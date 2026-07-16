"""
One-file experiment runner for electricity forecasting.

Main proposed model: VMD_TCN (VMD + TCN, fixed learning rate, no Optuna)
Baselines: GRU, LSTM, Transformer

Outputs:
  results_all_experiments/
    metrics/all_metrics.csv                 (per-seed metrics, incl. Seed column)
    metrics/metrics_aggregated.csv          (mean+/-std across seeds; >1 seed only)
    metrics/model_complexity.csv            (params + train time, for cost table)
    metrics/training_history.csv
    metrics/optuna_*_trials.csv
    predictions/<Model>/<Dataset>_<Building>_prediction_results.csv
    predictions/all_predictions_long.csv
    statistics/dm_tests.csv
    statistics/wilcoxon_error_tests.csv
    statistics/block_level_wilcoxon_tests.csv
    statistics/friedman_tests.csv
    statistics/nemenyi_<metric>.csv
    statistics/average_ranks_<metric>.csv
    statistics/absolute_error_long.csv
    statistics/plots/absolute_error_boxplot_all.png
    statistics/plots/absolute_error_violin_all.png
    config_used.json

Models (Dataset column): "test" (held-out test window of the 6 training
  buildings), "holdout" (Dec-2017 month of the same buildings), and "cross"
  (the unseen building "Hog") -> use the latter two for the generalization table.
Ablation (set run_ablation=True): adds "TCN" (raw signal, no VMD) alongside the
  main "VMD_TCN", to isolate the contribution of the VMD decomposition step.

Required packages:
  pip install numpy pandas scikit-learn scipy matplotlib torch optuna vmdpy

Run:
  python run_all_vmd_optuna_tcn_experiments.py

------------------------------------------------------------------------------
REPRODUCIBILITY & REPORTING NOTES (read before changing CONFIG)
------------------------------------------------------------------------------
Parity with the standalone scripts (combined_vmd_optuna_tcn.py / GRU.py /
LSTM.py / Transformer.py) is controlled by CONFIG and is ON by default:
  * reseed_per_model = True      -> each model is re-seeded right before it is
                                    built/trained, so GRU/LSTM/Transformer/VMD
                                    each start from the same RNG state they would
                                    have when run alone (not leftover state).
  * checkpoint_strategy = "final"-> save the last-epoch weights, like the
                                    standalone scripts (not a best-val checkpoint).
  * vmd.scale_fit_target =
        "full_series"            -> each VMD mode's StandardScaler is fit on the
                                    full mode series, exactly like the standalone.
  * vmd.eval_target = "modes_sum"-> VMD is scored against the summed-modes
                                    reconstruction, reproducing the standalone
                                    VMD numbers.
  * cuDNN benchmark is OFF (matches the standalone defaults; benchmark=True made
    the combined run drift nondeterministically).

IMPORTANT for the statistical tests (DM / Wilcoxon / Friedman / Nemenyi):
  With vmd.eval_target = "modes_sum", VMD is scored against an easier target than
  the GRU/LSTM/Transformer baselines (which are scored against the true series),
  so the cross-model significance tests are NOT apples-to-apples. For a fair,
  publication-grade comparison set vmd.eval_target = "original": every model is
  then scored against the same true series. This lowers VMD's headline numbers
  but makes the tests valid. Choose deliberately and state it in your methods.

The Diebold-Mariano implementation is CORRECT as written: it demeans d before
computing autocovariances, matching R forecast::dm.test. Do not remove the
centering (see the comment inside dm_test_errors for the full explanation).
------------------------------------------------------------------------------
"""

import os
import json
import time
import random
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from scipy.stats import t, wilcoxon, friedmanchisquare
try:
    from scipy.stats import studentized_range
except Exception:  # older scipy
    studentized_range = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def setup_plot_fonts() -> None:
    """Use Times New Roman everywhere if available, else a serif fallback.

    Looks for a 'Times New Roman' face; if the system only ships the
    metric-compatible 'Liberation Serif' / 'Nimbus Roman', those are used so
    figures still render in a Times-like serif on Linux servers.
    """
    preferred = ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in preferred if name in available), "serif")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [chosen],
        "mathtext.fontset": "stix",   # serif math to match Times
        "axes.unicode_minus": False,
    })
    print(f"[fonts] Matplotlib serif font set to: {chosen}")


try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception as exc:
    optuna = None
    OPTUNA_IMPORT_ERROR = exc

try:
    from vmdpy import VMD
except Exception as exc:
    VMD = None
    VMD_IMPORT_ERROR = exc

warnings.filterwarnings("ignore")

# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "meters", "cleaned")
ELECTRICITY_FILE = os.path.join(DATA_DIR, "electricity_cleaned.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results_all_experiments")

CONFIG: Dict[str, Any] = {
    "seed": 42,
    # >>> NEW: list of seeds for repeated runs. The full pipeline is run once per
    # seed and metrics are aggregated (mean +/- std) into metrics_aggregated.csv.
    # Q1 reviewers routinely require this. Use e.g. [42, 1, 2, 3, 4] for the paper.
    "seeds": [42],
    "seq_len": 96,                 # look-back window, hours
    "pred_len": 24,                # forecast horizon, hours
    "batch_size": 32,
    "inference_batch_size": 128,
    "epochs": 25,
    "learning_rate": 1e-4,
    "split_ratio": {"train": 0.7, "val": 0.1, "test": 0.2},
    "target_buildings": ["Wolf", "Bull", "Robin", "Fox", "Rat", "Eagle"],
    "test_buildings": ["Wolf", "Bull", "Robin", "Fox", "Rat", "Eagle", "Hog"],
    "holdout_start": "2017-12-01 00:00:00",
    "holdout_end": "2017-12-31 23:00:00",
    "hog_test_start": "2017-07-14 00:00:00",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "models_to_run": ["VMD_TCN", "GRU", "LSTM", "Transformer"],
    "reuse_existing_checkpoints": False,
    "save_long_prediction_file": True,

    # >>> NEW: ablation study. When True, an extra configuration is trained so you
    # can isolate the contribution of the VMD decomposition step for the paper's
    # ablation table: "TCN" (TCN on the raw signal, no VMD). The main model stays
    # "VMD_TCN" (VMD + TCN, fixed learning rate).
    "run_ablation": True,

    # >>> NEW: fair-comparison switch. When True, GRU/LSTM/Transformer/TCN each get
    # their own Optuna learning-rate search (same budget CONFIG["optuna"] defines),
    # so their comparison to the fixed-lr main model isn't handicapped by an
    # untuned LR. Leave False to reproduce the original fixed-lr baselines, but
    # state your choice in the paper. This has no effect on VMD_TCN, which always
    # uses the fixed CONFIG["learning_rate"] (no Optuna tuning anywhere).
    "tune_baselines": False,

    # >>> NEW: record per-model parameter count + train/inference time into
    # metrics/model_complexity.csv for the computational-cost table.
    "record_complexity": True,

    # --- Reproducibility / parity with the standalone scripts ---
    # The standalone scripts each call set_seed(42) once and then save the FINAL
    # epoch's weights. "final" reproduces that exactly. "best_val" keeps the
    # validation-based early checkpoint that the original runner used.
    "checkpoint_strategy": "final",   # options: final, best_val
    # Re-seed right before each model is built/trained so every model starts from
    # the same RNG state it would have when run alone. With False, only the first
    # model matches its standalone counterpart.
    "reseed_per_model": True,

    # VMD parameters from the provided proposed model script.
    "vmd": {
        "alpha": 1000,
        "tau": 0,
        "K": 6,
        "DC": 0,
        "init": 1,
        "tol": 1e-7,
        # >>> CHANGED to "train_only" (was "full_series"). "full_series" fits each
        # mode's StandardScaler using the WHOLE series including the test period,
        # which leaks test statistics into training normalisation. "train_only" is
        # leakage-free and is the correct choice for a publishable result. Switch
        # back only if you must exactly reproduce the old standalone numbers.
        "scale_fit_target": "train_only",   # options: train_only, full_series
        "scale_fit_unseen": "own_full",     # options: own_full
        # >>> CHANGED to "original" (was "modes_sum"). "modes_sum" scores the
        # proposed model against the sum-of-modes reconstruction -- an EASIER
        # target than the true series that the GRU/LSTM/Transformer baselines are
        # scored against -- so the comparison and all significance tests were not
        # apples-to-apples. "original" scores every model against the same true
        # cleaned series. This lowers the proposed model's headline numbers but is
        # the only fair, defensible setting. State it explicitly in your methods.
        "eval_target": "original",          # options: original, modes_sum
        # >>> NEW (documentation only; default keeps current behaviour).
        # "global": VMD is computed once on the full series, then windowed. This is
        #   fast but NON-CAUSAL: each mode value uses information from the entire
        #   series (including the future), so test inputs contain look-ahead. This
        #   is the single most common reason VMD-hybrid papers are rejected.
        # "causal": (recommended for the final paper) re-decompose using only data
        #   up to each forecast origin. It is much slower. See the CAUSAL VMD note
        #   near prepare_vmd_bundle() for the exact change required to enable it.
        "decompose_mode": "causal",         # options: global, causal(see note)
    },

    "optuna": {
        "n_trials": 10,
        "objective_epochs": 3,
        "lr_low": 1e-4,
        "lr_high": 1e-2,
    },

    "tcn": {
        "num_channels": [16, 32, 64],
        "kernel_size": 3,
        "dropout": 0.1,
    },

    "gru": {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.1,
    },

    "lstm": {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.1,
    },

    "transformer": {
        "d_model": 64,
        "nhead": 4,
        "num_encoder_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
    },

    "statistics": {
        "proposed_model": "VMD_TCN",
        "dm_loss": ["squared", "absolute"],
        # We compare the continuous reconstructed prediction series, so h=1 is appropriate.
        # If you compare overlapping 24-step windows directly, change this to 24.
        "dm_horizon": 1,
        "alpha": 0.05,
        "metrics_for_block_tests": ["MAE", "RMSE", "MAPE", "SMAPE"],
        "max_plot_points_per_model": 20000,
    },
}

DIRS = {
    "models": os.path.join(OUTPUT_DIR, "models"),
    "predictions": os.path.join(OUTPUT_DIR, "predictions"),
    "metrics": os.path.join(OUTPUT_DIR, "metrics"),
    "statistics": os.path.join(OUTPUT_DIR, "statistics"),
    "plots": os.path.join(OUTPUT_DIR, "statistics", "plots"),
}

# ============================================================
# 2. UTILITIES
# ============================================================
def ensure_dirs() -> None:
    for path in DIRS.values():
        os.makedirs(path, exist_ok=True)
    for model_name in CONFIG["models_to_run"]:
        os.makedirs(os.path.join(DIRS["models"], model_name), exist_ok=True)
        os.makedirs(os.path.join(DIRS["predictions"], model_name), exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # The standalone scripts leave cuDNN at its library defaults. benchmark=True
    # picks algorithms nondeterministically and made the combined run drift away
    # from the standalone results, so keep it off here.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clean_flat_sequences(series: np.ndarray, window_size: int = 5, threshold: float = 1e-5) -> np.ndarray:
    """Replace flat-line segments with interpolated values."""
    s = pd.Series(series).astype(float)
    rolling_std = s.rolling(window=window_size).std()
    mask = rolling_std < threshold
    s[mask] = np.nan
    s = s.interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")
    return s.values.astype(float)


def find_building_column(df: pd.DataFrame, building: str) -> Optional[str]:
    matches = [c for c in df.columns if building.lower() in c.lower()]
    return matches[0] if matches else None


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-5) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)


def safe_smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-5) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": rmse,
        "MAPE": safe_mape(y_true, y_pred),
        "SMAPE": safe_smape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
    }


def continuous_from_windows(windows: np.ndarray) -> np.ndarray:
    """Convert [n_windows, pred_len] forecasts into one continuous series."""
    if windows.ndim != 2:
        raise ValueError("windows must be a 2D array [n_windows, pred_len].")
    if windows.shape[0] == 0:
        return np.array([], dtype=float)
    first_horizon = windows[:, 0]
    tail = windows[-1, 1:]
    return np.concatenate([first_horizon, tail]).astype(float)


def save_prediction_csv(model_name: str, dataset_name: str, building: str, df: pd.DataFrame) -> str:
    model_dir = os.path.join(DIRS["predictions"], model_name)
    os.makedirs(model_dir, exist_ok=True)
    filename = f"{dataset_name}_{building}_prediction_results.csv"
    path = os.path.join(model_dir, filename)
    df.to_csv(path, index=False)
    return path


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


# >>> NEW: model-complexity recording for the computational-cost table.
COMPLEXITY_ROWS: List[Dict[str, Any]] = []


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def record_complexity(model_name: str, model: nn.Module, training_time: float,
                      best_lr: float, seed: int) -> None:
    if not CONFIG.get("record_complexity", True):
        return
    COMPLEXITY_ROWS.append({
        "Seed": seed,
        "Model": model_name,
        "Trainable_Params": count_parameters(model),
        "Training_Time_sec": round(float(training_time), 4),
        "Learning_Rate": float(best_lr),
        "Epochs": CONFIG["epochs"],
        "Device": CONFIG["device"],
    })


# ============================================================
# 3. DATASET
# ============================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data = torch.FloatTensor(np.asarray(data, dtype=float))
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return x, y


def make_dataset(data: np.ndarray, seq_len: int, pred_len: int, name: str) -> TimeSeriesDataset:
    ds = TimeSeriesDataset(data, seq_len, pred_len)
    if len(ds) == 0:
        raise ValueError(
            f"Not enough samples for {name}. Need at least seq_len + pred_len = "
            f"{seq_len + pred_len}, got {len(data)}."
        )
    return ds


# ============================================================
# 4. MODELS
# ============================================================
class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.relu(self.conv(x)))


class SimpleTCN(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_channels: Optional[List[int]] = None,
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        if num_channels is None:
            num_channels = [16, 32, 64]
        layers = []
        in_channels = 1
        temp_len = seq_len
        for i, out_channels in enumerate(num_channels):
            dilation_size = 2 ** i
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation=dilation_size, dropout=dropout))
            in_channels = out_channels
            temp_len = temp_len - dilation_size * (kernel_size - 1)
        if temp_len <= 0:
            raise ValueError("TCN output length became <= 0. Reduce kernel_size/channels or increase seq_len.")
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1] * temp_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)              # [batch, 1, seq_len]
        out = self.network(x)
        out = out.reshape(out.size(0), -1)
        return self.fc(out)


class GRUModel(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.fc(last)


class LSTMModel(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embedding


class TransformerModel(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, d_model: int = 64, nhead: int = 4,
                 num_encoder_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model * seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


def build_model(model_name: str) -> nn.Module:
    seq_len = CONFIG["seq_len"]
    pred_len = CONFIG["pred_len"]
    # "TCN" is the raw-signal ablation: same TCN as the main "VMD_TCN" model but
    # trained directly on the un-decomposed series (no VMD).
    if model_name in ("VMD_TCN", "TCN"):
        tcfg = CONFIG["tcn"]
        return SimpleTCN(seq_len, pred_len, tcfg["num_channels"], tcfg["kernel_size"], tcfg["dropout"])
    if model_name == "GRU":
        cfg = CONFIG["gru"]
        return GRUModel(seq_len, pred_len, cfg["hidden_size"], cfg["num_layers"], cfg["dropout"])
    if model_name == "LSTM":
        cfg = CONFIG["lstm"]
        return LSTMModel(seq_len, pred_len, cfg["hidden_size"], cfg["num_layers"], cfg["dropout"])
    if model_name == "Transformer":
        cfg = CONFIG["transformer"]
        return TransformerModel(
            seq_len, pred_len,
            cfg["d_model"], cfg["nhead"], cfg["num_encoder_layers"],
            cfg["dim_feedforward"], cfg["dropout"]
        )
    raise ValueError(f"Unknown model_name: {model_name}")


# ============================================================
# 5. TRAINING AND VALIDATION
# ============================================================
def train_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
    return total_loss / max(1, len(loader))


def train_single_model(model_name: str, train_loader: DataLoader, val_loader: DataLoader,
                       lr: float, model_path: str) -> Tuple[nn.Module, List[Dict[str, Any]], float]:
    device = torch.device(CONFIG["device"])
    model = build_model(model_name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = []
    best_val = float("inf")
    strategy = CONFIG.get("checkpoint_strategy", "final")
    start = time.time()

    print(f"\n[{now_str()}] Training {model_name} | lr={lr:.8f} | checkpoint={strategy}")
    for epoch in range(CONFIG["epochs"]):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
        # "best_val" keeps the lowest-validation checkpoint (original runner).
        # "final" mirrors the standalone scripts, which save after the last epoch.
        if strategy == "best_val" and improved:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
        history.append({
            "Model": model_name,
            "Epoch": epoch + 1,
            "Train_Loss": tr_loss,
            "Val_Loss": val_loss,
            "Best_Val_Loss": best_val,
            "Saved_Best": int(improved),
        })
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == CONFIG["epochs"]:
            print(f"  Epoch {epoch+1:03d}/{CONFIG['epochs']} | train={tr_loss:.6f} | val={val_loss:.6f}")

    training_time = time.time() - start
    if strategy != "best_val":
        # Save (and then reload) the final-epoch weights, exactly like the standalone.
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[{now_str()}] {model_name} training completed in {training_time:.2f}s | best_val={best_val:.6f}")
    return model, history, training_time


# >>> NEW: generic LR tuner so GRU/LSTM/Transformer/TCN can be Optuna-tuned with
# the SAME budget CONFIG["optuna"] defines when CONFIG["tune_baselines"] is True.
# VMD_TCN itself is never Optuna-tuned; it always trains with the fixed
# CONFIG["learning_rate"] (see run_vmd_tcn below).
def optuna_objective_generic(trial, model_name: str, train_loader: DataLoader, val_loader: DataLoader) -> float:
    if optuna is None:
        raise ImportError(f"Optuna is required. Install with: pip install optuna. Original error: {OPTUNA_IMPORT_ERROR}")
    device = torch.device(CONFIG["device"])
    lr = trial.suggest_float("lr", CONFIG["optuna"]["lr_low"], CONFIG["optuna"]["lr_high"], log=True)
    model = build_model(model_name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    try:
        for _ in range(CONFIG["optuna"]["objective_epochs"]):
            train_epoch(model, train_loader, optimizer, criterion, device)
        return validate(model, val_loader, criterion, device)
    except Exception:
        return float("inf")


def tune_model_lr(model_name: str, train_loader: DataLoader, val_loader: DataLoader) -> float:
    if optuna is None:
        raise ImportError(f"Optuna is required to tune baselines. Install with: pip install optuna. Original error: {OPTUNA_IMPORT_ERROR}")
    print(f"\n[{now_str()}] Optuna search for {model_name} learning rate")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: optuna_objective_generic(trial, model_name, train_loader, val_loader),
                   n_trials=CONFIG["optuna"]["n_trials"])
    best_lr = float(study.best_params["lr"])
    trials_path = os.path.join(DIRS["metrics"], f"optuna_{model_name}_trials.csv")
    study.trials_dataframe().to_csv(trials_path, index=False)
    print(f"  {model_name} best LR = {best_lr:.8f} | trials saved to {trials_path}")
    return best_lr


# ============================================================
# 6. DATA PREPARATION
# ============================================================
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(ELECTRICITY_FILE):
        raise FileNotFoundError(
            f"Cannot find data file: {ELECTRICITY_FILE}\n"
            f"Expected path style: BASE_DIR/meters/cleaned/electricity_cleaned.csv"
        )
    df_all = pd.read_csv(ELECTRICITY_FILE)
    if "timestamp" not in df_all.columns:
        raise ValueError("The CSV must contain a 'timestamp' column.")
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])

    holdout_mask = (
        (df_all["timestamp"] >= pd.to_datetime(CONFIG["holdout_start"])) &
        (df_all["timestamp"] <= pd.to_datetime(CONFIG["holdout_end"]))
    )
    df_holdout = df_all[holdout_mask].reset_index(drop=True)
    df_main = df_all[~holdout_mask].reset_index(drop=True)

    print(f"Data loaded: {ELECTRICITY_FILE}")
    print(f"Main data      : {df_main['timestamp'].min()} -> {df_main['timestamp'].max()} ({len(df_main)} rows)")
    print(f"Holdout period : {df_holdout['timestamp'].min()} -> {df_holdout['timestamp'].max()} ({len(df_holdout)} rows)")
    return df_all, df_main, df_holdout


def prepare_raw_bundle(df_main: pd.DataFrame) -> Dict[str, Any]:
    seq_len = CONFIG["seq_len"]
    pred_len = CONFIG["pred_len"]
    target_buildings = CONFIG["target_buildings"]
    split_ratio = CONFIG["split_ratio"]

    building_data: Dict[str, Dict[str, Any]] = {}
    all_raw_train = []

    print(f"\n[{now_str()}] Preparing raw combined data for GRU/LSTM/Transformer")
    for bid in target_buildings:
        col_name = find_building_column(df_main, bid)
        if col_name is None:
            print(f"  Skip {bid}: column not found")
            continue
        raw = df_main[col_name].values
        series = clean_flat_sequences(raw)
        total_len = len(series)
        train_len = int(total_len * split_ratio["train"])
        val_len = int(total_len * split_ratio["val"])
        building_data[bid] = {
            "col_name": col_name,
            "series": series,
            "train_len": train_len,
            "val_len": val_len,
        }
        all_raw_train.append(series[:train_len])
        print(f"  {bid}: total={total_len}, train={train_len}, val={val_len}, test={total_len - train_len - val_len}")

    if not building_data:
        raise ValueError("No target building columns were found in df_main.")

    scaler = StandardScaler()
    scaler.fit(np.concatenate(all_raw_train).reshape(-1, 1))

    train_datasets = []
    val_datasets = []
    for bid, info in building_data.items():
        scaled = scaler.transform(info["series"].reshape(-1, 1)).flatten()
        info["series_scaled"] = scaled
        train_data = scaled[:info["train_len"]]
        val_data = scaled[info["train_len"]: info["train_len"] + info["val_len"]]
        train_datasets.append(make_dataset(train_data, seq_len, pred_len, f"{bid} raw train"))
        val_datasets.append(make_dataset(val_data, seq_len, pred_len, f"{bid} raw val"))

    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=CONFIG["batch_size"], shuffle=False)

    print(f"  Combined raw train samples: {len(train_loader.dataset)}")
    print(f"  Combined raw val samples  : {len(val_loader.dataset)}")
    return {
        "building_data": building_data,
        "scaler": scaler,
        "train_loader": train_loader,
        "val_loader": val_loader,
    }


def prepare_vmd_bundle(df_main: pd.DataFrame) -> Dict[str, Any]:
    if VMD is None:
        raise ImportError(f"vmdpy is required for VMD_TCN. Install with: pip install vmdpy. Original error: {VMD_IMPORT_ERROR}")

    seq_len = CONFIG["seq_len"]
    pred_len = CONFIG["pred_len"]
    vcfg = CONFIG["vmd"]
    target_buildings = CONFIG["target_buildings"]
    split_ratio = CONFIG["split_ratio"]

    building_data: Dict[str, Dict[str, Any]] = {}
    train_datasets = []
    val_datasets = []

    # >>> NON-CAUSALITY WARNING + CAUSAL RECIPE -------------------------------
    # The loop below calls VMD(series, ...) on the FULL main series and then
    # windows it. Each mode value therefore depends on the entire series,
    # including future samples -> the 96-hour test inputs contain look-ahead.
    # This is the #1 reason VMD-hybrid forecasting papers are rejected.
    #
    # To make decomposition CAUSAL (recommended for the final paper), replace the
    # global decomposition with an expanding/rolling one at inference time:
    #   for each forecast origin t (predicting t+1..t+24):
    #       modes_t, _, _ = VMD(series[:t+1], alpha, tau, K, DC, init, tol)
    #       window = modes_t[:, t+1-seq_len : t+1]   # last seq_len columns
    #       feed `window` (per mode) to the trained TCN
    # i.e. fit VMD only on data up to t. This costs one VMD call per origin (slow)
    # but removes the leakage. Training modes should likewise be derived from the
    # training split only. Validate on one building first; it is much slower.
    if CONFIG["vmd"].get("decompose_mode", "global") == "global":
        print("  [WARNING] VMD decompose_mode='global' is NON-CAUSAL (uses the full "
              "series, including the test period). Acceptable for a quick run, but "
              "set decompose_mode='causal' (see recipe in prepare_vmd_bundle) for a "
              "leakage-free result before submitting to a Q1 journal.")
    # ------------------------------------------------------------------------

    print(f"\n[{now_str()}] Preparing VMD modes for VMD_TCN")
    for bid in target_buildings:
        col_name = find_building_column(df_main, bid)
        if col_name is None:
            print(f"  Skip {bid}: column not found")
            continue

        raw = df_main[col_name].values
        series = clean_flat_sequences(raw)
        total_len = len(series)
        train_len = int(total_len * split_ratio["train"])
        val_len = int(total_len * split_ratio["val"])

        print(f"  {bid}: running VMD K={vcfg['K']} on {total_len} samples")
        modes, _, _ = VMD(series, vcfg["alpha"], vcfg["tau"], vcfg["K"], vcfg["DC"], vcfg["init"], vcfg["tol"])
        n_modes = modes.shape[0]

        scalers = []
        scaled_modes = []
        for mode_idx in range(n_modes):
            scaler = StandardScaler()
            mode = modes[mode_idx].reshape(-1, 1)
            if vcfg["scale_fit_target"] == "train_only":
                scaler.fit(mode[:train_len])
            elif vcfg["scale_fit_target"] == "full_series":
                scaler.fit(mode)
            else:
                raise ValueError("CONFIG['vmd']['scale_fit_target'] must be 'train_only' or 'full_series'.")
            mode_scaled = scaler.transform(mode).flatten()
            scalers.append(scaler)
            scaled_modes.append(mode_scaled)

            train_mode = mode_scaled[:train_len]
            val_mode = mode_scaled[train_len: train_len + val_len]
            train_datasets.append(make_dataset(train_mode, seq_len, pred_len, f"{bid} VMD mode {mode_idx+1} train"))
            val_datasets.append(make_dataset(val_mode, seq_len, pred_len, f"{bid} VMD mode {mode_idx+1} val"))

        building_data[bid] = {
            "col_name": col_name,
            "series": series,
            "modes": modes,
            "scaled_modes": scaled_modes,
            "scalers": scalers,
            "n_modes": n_modes,
            "train_len": train_len,
            "val_len": val_len,
        }
        print(f"  {bid}: VMD done -> {n_modes} modes")

    if not building_data:
        raise ValueError("No target building VMD data could be prepared.")

    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=CONFIG["batch_size"], shuffle=False)
    print(f"  Combined VMD train samples: {len(train_loader.dataset)}")
    print(f"  Combined VMD val samples  : {len(val_loader.dataset)}")
    return {
        "building_data": building_data,
        "train_loader": train_loader,
        "val_loader": val_loader,
    }


# ============================================================
# 7. INFERENCE
# ============================================================
def predict_raw_series(model: nn.Module, scaler: StandardScaler, series: np.ndarray, timestamps: pd.Series,
                       model_name: str, dataset_name: str, building: str) -> pd.DataFrame:
    seq_len = CONFIG["seq_len"]
    pred_len = CONFIG["pred_len"]
    device = torch.device(CONFIG["device"])

    scaled = scaler.transform(series.reshape(-1, 1)).flatten()
    ds = make_dataset(scaled, seq_len, pred_len, f"{model_name} {dataset_name} {building}")
    loader = DataLoader(ds, batch_size=CONFIG["inference_batch_size"], shuffle=False)

    preds_windows = []
    with torch.no_grad():
        for x, _ in loader:
            out = model(x.to(device))
            pred = scaler.inverse_transform(out.cpu().numpy().reshape(-1, 1)).reshape(-1, pred_len)
            preds_windows.append(pred)
    preds_windows_arr = np.concatenate(preds_windows, axis=0)
    pred_series = continuous_from_windows(preds_windows_arr)

    actual_series = series[seq_len: seq_len + len(pred_series)].astype(float)
    ts = pd.Series(timestamps).iloc[seq_len: seq_len + len(pred_series)].values

    return pd.DataFrame({
        "Timestamp": ts,
        "Actual": actual_series,
        "Prediction": pred_series,
        "Model": model_name,
        "Dataset": dataset_name,
        "Building": building,
    })


def predict_vmd_series(model: nn.Module, modes: np.ndarray, scalers: List[StandardScaler], original_series: np.ndarray,
                       timestamps: pd.Series, model_name: str, dataset_name: str, building: str) -> pd.DataFrame:
    seq_len = CONFIG["seq_len"]
    pred_len = CONFIG["pred_len"]
    device = torch.device(CONFIG["device"])

    final_pred_series: Optional[np.ndarray] = None
    for mode_idx in range(modes.shape[0]):
        mode = modes[mode_idx]
        scaler = scalers[mode_idx]
        scaled = scaler.transform(mode.reshape(-1, 1)).flatten()
        ds = make_dataset(scaled, seq_len, pred_len, f"{model_name} {dataset_name} {building} mode {mode_idx+1}")
        loader = DataLoader(ds, batch_size=CONFIG["inference_batch_size"], shuffle=False)

        mode_pred_windows = []
        with torch.no_grad():
            for x, _ in loader:
                out = model(x.to(device))
                pred = scaler.inverse_transform(out.cpu().numpy().reshape(-1, 1)).reshape(-1, pred_len)
                mode_pred_windows.append(pred)
        mode_pred_windows_arr = np.concatenate(mode_pred_windows, axis=0)
        mode_pred_series = continuous_from_windows(mode_pred_windows_arr)
        if final_pred_series is None:
            final_pred_series = np.zeros_like(mode_pred_series, dtype=float)
        min_len = min(len(final_pred_series), len(mode_pred_series))
        final_pred_series = final_pred_series[:min_len] + mode_pred_series[:min_len]

    if final_pred_series is None:
        raise RuntimeError(f"No VMD predictions produced for {building} {dataset_name}.")

    n = len(final_pred_series)
    eval_target = CONFIG["vmd"].get("eval_target", "modes_sum")
    if eval_target == "modes_sum":
        # Standalone parity: ground truth is the sum of the (inverse-transformed)
        # mode targets, i.e. the VMD reconstruction over the forecast window.
        reconstruction = modes.sum(axis=0)
        actual_series = reconstruction[seq_len: seq_len + n].astype(float)
    elif eval_target == "original":
        # Stricter / fairer cross-model target: the true cleaned series, which
        # also contains the VMD reconstruction residual.
        actual_series = original_series[seq_len: seq_len + n].astype(float)
    else:
        raise ValueError("CONFIG['vmd']['eval_target'] must be 'modes_sum' or 'original'.")
    ts = pd.Series(timestamps).iloc[seq_len: seq_len + n].values

    return pd.DataFrame({
        "Timestamp": ts,
        "Actual": actual_series,
        "Prediction": final_pred_series,
        "Model": model_name,
        "Dataset": dataset_name,
        "Building": building,
    })


def evaluate_and_store_prediction(pred_df: pd.DataFrame, training_time: float, testing_time: float) -> Dict[str, Any]:
    metrics = compute_metrics(pred_df["Actual"].values, pred_df["Prediction"].values)
    row = {
        "Model": pred_df["Model"].iloc[0],
        "Dataset": pred_df["Dataset"].iloc[0],
        "Building": pred_df["Building"].iloc[0],
        **metrics,
        "Training_Time_sec": round(training_time, 4),
        "Testing_Time_sec": round(testing_time, 4),
        "N_Points": int(len(pred_df)),
    }
    save_prediction_csv(row["Model"], row["Dataset"], row["Building"], pred_df)
    print(
        f"  {row['Model']} | {row['Dataset']} | {row['Building']} | "
        f"MAE={row['MAE']:.4f} RMSE={row['RMSE']:.4f} MAPE={row['MAPE']:.4f}% SMAPE={row['SMAPE']:.4f}% R2={row['R2']:.4f}"
    )
    return row


# ============================================================
# 8. RUN MODEL GROUPS
# ============================================================
def run_baseline_model(model_name: str, raw_bundle: Dict[str, Any], df_all: pd.DataFrame,
                       df_main: pd.DataFrame, df_holdout: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[Tuple[str, str, str], pd.DataFrame]]:
    # Each standalone script seeds once and then builds/trains exactly one model.
    # Re-seeding here gives this model the same initialisation and shuffling
    # stream it would get on its own, instead of inheriting the RNG state left
    # behind by the previously trained models.
    if CONFIG.get("reseed_per_model", True):
        set_seed(CONFIG["seed"])

    model_dir = os.path.join(DIRS["models"], model_name)
    model_path = os.path.join(model_dir, f"{model_name}_combined.pt")

    if CONFIG["reuse_existing_checkpoints"] and os.path.exists(model_path):
        device = torch.device(CONFIG["device"])
        model = build_model(model_name).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        history = []
        training_time = 0.0
        print(f"\nLoaded existing checkpoint for {model_name}: {model_path}")
    else:
        # Fair comparison: optionally Optuna-tune this baseline's LR with the same
        # budget as the proposed model. Otherwise use the fixed CONFIG learning_rate.
        if CONFIG.get("tune_baselines", False):
            lr = tune_model_lr(model_name, raw_bundle["train_loader"], raw_bundle["val_loader"])
        else:
            lr = CONFIG["learning_rate"]
        model, history, training_time = train_single_model(
            model_name,
            raw_bundle["train_loader"],
            raw_bundle["val_loader"],
            lr,
            model_path,
        )
        for row in history:
            row["Learning_Rate"] = lr
        record_complexity(model_name, model, training_time, lr, CONFIG["seed"])

    metrics_rows = []
    predictions: Dict[Tuple[str, str, str], pd.DataFrame] = {}
    scaler = raw_bundle["scaler"]
    building_data = raw_bundle["building_data"]

    print(f"\n[{now_str()}] Testing {model_name}")
    for bid, info in building_data.items():
        train_len = info["train_len"]
        val_len = info["val_len"]
        segment = info["series"][train_len + val_len:]
        segment_ts = df_main["timestamp"].iloc[train_len + val_len:].reset_index(drop=True)
        t0 = time.time()
        pred_df = predict_raw_series(model, scaler, segment, segment_ts, model_name, "test", bid)
        testing_time = time.time() - t0
        metrics_rows.append(evaluate_and_store_prediction(pred_df, training_time, testing_time))
        predictions[(model_name, "test", bid)] = pred_df

        if len(df_holdout) > 0:
            holdout_col = find_building_column(df_holdout, bid)
            if holdout_col is not None:
                holdout_series = clean_flat_sequences(df_holdout[holdout_col].values)
                holdout_ts = df_holdout["timestamp"].reset_index(drop=True)
                t0 = time.time()
                h_pred_df = predict_raw_series(model, scaler, holdout_series, holdout_ts, model_name, "holdout", bid)
                testing_time = time.time() - t0
                metrics_rows.append(evaluate_and_store_prediction(h_pred_df, training_time, testing_time))
                predictions[(model_name, "holdout", bid)] = h_pred_df

    extra_test_buildings = [b for b in CONFIG["test_buildings"] if b not in CONFIG["target_buildings"]]
    for test_bid in extra_test_buildings:
        test_col = find_building_column(df_all, test_bid)
        if test_col is None:
            print(f"  Skip cross-building {test_bid}: column not found")
            continue
        mask = df_all["timestamp"] >= pd.to_datetime(CONFIG["hog_test_start"])
        df_hog = df_all[mask].reset_index(drop=True)
        hog_series = clean_flat_sequences(df_hog[test_col].values)
        hog_ts = df_hog["timestamp"].reset_index(drop=True)
        t0 = time.time()
        hog_pred_df = predict_raw_series(model, scaler, hog_series, hog_ts, model_name, "cross", test_bid)
        testing_time = time.time() - t0
        metrics_rows.append(evaluate_and_store_prediction(hog_pred_df, training_time, testing_time))
        predictions[(model_name, "cross", test_bid)] = hog_pred_df

    return metrics_rows, history, predictions


def run_vmd_tcn(vmd_bundle: Dict[str, Any], df_all: pd.DataFrame,
                df_main: pd.DataFrame, df_holdout: pd.DataFrame,
                model_name: str = "VMD_TCN") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[Tuple[str, str, str], pd.DataFrame]]:
    # Same rationale as the baselines: start training from a fresh seed so this
    # matches the standalone VMD script's RNG state.
    if CONFIG.get("reseed_per_model", True):
        set_seed(CONFIG["seed"])
    model_dir = os.path.join(DIRS["models"], model_name)
    model_path = os.path.join(model_dir, f"{model_name}_combined.pt")

    if CONFIG["reuse_existing_checkpoints"] and os.path.exists(model_path):
        device = torch.device(CONFIG["device"])
        model = build_model(model_name).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        history = []
        training_time = 0.0
        print(f"\nLoaded existing checkpoint for {model_name}: {model_path}")
    else:
        # VMD_TCN: VMD + TCN with a fixed learning rate (no Optuna tuning).
        lr = CONFIG["learning_rate"]
        model, history, training_time = train_single_model(
            model_name,
            vmd_bundle["train_loader"],
            vmd_bundle["val_loader"],
            lr,
            model_path,
        )
        for row in history:
            row["Learning_Rate"] = lr
        record_complexity(model_name, model, training_time, lr, CONFIG["seed"])

    metrics_rows = []
    predictions: Dict[Tuple[str, str, str], pd.DataFrame] = {}
    building_data = vmd_bundle["building_data"]
    vcfg = CONFIG["vmd"]

    print(f"\n[{now_str()}] Testing {model_name}")
    for bid, info in building_data.items():
        train_len = info["train_len"]
        val_len = info["val_len"]
        mode_segment = info["modes"][:, train_len + val_len:]
        original_segment = info["series"][train_len + val_len:]
        segment_ts = df_main["timestamp"].iloc[train_len + val_len:].reset_index(drop=True)
        t0 = time.time()
        pred_df = predict_vmd_series(model, mode_segment, info["scalers"], original_segment, segment_ts, model_name, "test", bid)
        testing_time = time.time() - t0
        metrics_rows.append(evaluate_and_store_prediction(pred_df, training_time, testing_time))
        predictions[(model_name, "test", bid)] = pred_df

        if len(df_holdout) > 0:
            holdout_col = find_building_column(df_holdout, bid)
            if holdout_col is not None:
                holdout_series = clean_flat_sequences(df_holdout[holdout_col].values)
                print(f"  {model_name} | holdout | {bid}: running VMD")
                holdout_modes, _, _ = VMD(holdout_series, vcfg["alpha"], vcfg["tau"], vcfg["K"], vcfg["DC"], vcfg["init"], vcfg["tol"])
                if holdout_modes.shape[0] == len(info["scalers"]):
                    holdout_ts = df_holdout["timestamp"].reset_index(drop=True)
                    t0 = time.time()
                    h_pred_df = predict_vmd_series(model, holdout_modes, info["scalers"], holdout_series, holdout_ts, model_name, "holdout", bid)
                    testing_time = time.time() - t0
                    metrics_rows.append(evaluate_and_store_prediction(h_pred_df, training_time, testing_time))
                    predictions[(model_name, "holdout", bid)] = h_pred_df
                else:
                    print(f"  Skip holdout {bid}: VMD mode count mismatch")

    extra_test_buildings = [b for b in CONFIG["test_buildings"] if b not in CONFIG["target_buildings"]]
    for test_bid in extra_test_buildings:
        test_col = find_building_column(df_all, test_bid)
        if test_col is None:
            print(f"  Skip cross-building {test_bid}: column not found")
            continue
        mask = df_all["timestamp"] >= pd.to_datetime(CONFIG["hog_test_start"])
        df_hog = df_all[mask].reset_index(drop=True)
        hog_series = clean_flat_sequences(df_hog[test_col].values)
        print(f"  {model_name} | cross | {test_bid}: running VMD")
        hog_modes, _, _ = VMD(hog_series, vcfg["alpha"], vcfg["tau"], vcfg["K"], vcfg["DC"], vcfg["init"], vcfg["tol"])

        # Reproduce the previous Hog-own-scaler logic for the unseen building.
        hog_scalers = []
        for mode_idx in range(hog_modes.shape[0]):
            sc = StandardScaler()
            sc.fit(hog_modes[mode_idx].reshape(-1, 1))
            hog_scalers.append(sc)

        hog_ts = df_hog["timestamp"].reset_index(drop=True)
        t0 = time.time()
        hog_pred_df = predict_vmd_series(model, hog_modes, hog_scalers, hog_series, hog_ts, model_name, "cross", test_bid)
        testing_time = time.time() - t0
        metrics_rows.append(evaluate_and_store_prediction(hog_pred_df, training_time, testing_time))
        predictions[(model_name, "cross", test_bid)] = hog_pred_df

    return metrics_rows, history, predictions


# ============================================================
# 9. STATISTICAL TESTS
# ============================================================
def dm_test_errors(e_proposed: np.ndarray, e_baseline: np.ndarray, loss: str = "squared", h: int = 1) -> Dict[str, Any]:
    """
    Diebold-Mariano test.
    d_t = baseline_loss_t - proposed_loss_t.
    Positive mean_d means the proposed model has lower loss.
    """
    e_proposed = np.asarray(e_proposed, dtype=float).reshape(-1)
    e_baseline = np.asarray(e_baseline, dtype=float).reshape(-1)
    n = min(len(e_proposed), len(e_baseline))
    e_proposed = e_proposed[:n]
    e_baseline = e_baseline[:n]
    if n < 5:
        return {"DM_Statistic": np.nan, "P_Value": np.nan, "Mean_Loss_Diff_BaselineMinusProposed": np.nan, "N": n}

    if loss == "squared":
        loss_prop = e_proposed ** 2
        loss_base = e_baseline ** 2
    elif loss == "absolute":
        loss_prop = np.abs(e_proposed)
        loss_base = np.abs(e_baseline)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'.")

    d = loss_base - loss_prop
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 5:
        return {"DM_Statistic": np.nan, "P_Value": np.nan, "Mean_Loss_Diff_BaselineMinusProposed": np.nan, "N": n}
    mean_d = float(np.mean(d))
    h = max(1, min(int(h), n - 1))

    # NOTE ON CENTERING (do not "fix" this by removing it):
    # The DM long-run variance is the sum of AUTOCOVARIANCES of d, and an
    # autocovariance is defined on the demeaned series. This matches the
    # canonical implementation R forecast::dm.test, which uses
    # acf(d, type="covariance") -- and R's acf demeans by default. At h=1 this
    # reduces exactly to the paired/one-sample t-statistic on d. Using the raw
    # (uncentered) series instead inflates the variance by mean_d**2 and caps the
    # statistic near sqrt(n), which destroys the test's power precisely when the
    # models differ most. Keep the centering.
    d_centered = d - mean_d

    # Autocovariances use the 1/n divisor for every lag (R's acf convention),
    # not 1/(n-lag). This only matters when h > 1; at h=1 the loop is skipped.
    gamma0 = float(np.sum(d_centered * d_centered) / n)
    long_run_var = gamma0
    for lag in range(1, h):
        cov = float(np.sum(d_centered[lag:] * d_centered[:-lag]) / n)
        long_run_var += 2.0 * cov

    if long_run_var <= 0 or not np.isfinite(long_run_var):
        return {"DM_Statistic": np.nan, "P_Value": np.nan, "Mean_Loss_Diff_BaselineMinusProposed": mean_d, "N": n}

    dm_stat = mean_d / np.sqrt(long_run_var / n)
    # Harvey-Leybourne-Newbold small sample correction.
    correction = np.sqrt((n + 1 - 2 * h + (h * (h - 1) / n)) / n)
    dm_adj = float(dm_stat * correction)
    p_value = float(1.0 - t.cdf(dm_adj, df=n - 1))
    return {
        "DM_Statistic": dm_adj,
        "P_Value": p_value,
        "Mean_Loss_Diff_BaselineMinusProposed": mean_d,
        "N": n,
    }


def aligned_errors(proposed_df: pd.DataFrame, baseline_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prop = proposed_df[["Timestamp", "Actual", "Prediction"]].copy()
    base = baseline_df[["Timestamp", "Prediction"]].copy()
    prop = prop.rename(columns={"Prediction": "Prediction_Proposed"})
    base = base.rename(columns={"Prediction": "Prediction_Baseline"})
    merged = pd.merge(prop, base, on="Timestamp", how="inner")
    y = merged["Actual"].values.astype(float)
    e_prop = y - merged["Prediction_Proposed"].values.astype(float)
    e_base = y - merged["Prediction_Baseline"].values.astype(float)
    return y, e_prop, e_base


def run_error_sequence_tests(predictions: Dict[Tuple[str, str, str], pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    proposed_model = CONFIG["statistics"]["proposed_model"]
    alpha = CONFIG["statistics"]["alpha"]
    dm_rows = []
    wilcoxon_rows = []
    error_rows = []

    keys = sorted(predictions.keys(), key=lambda x: (x[1], x[2], x[0]))
    baseline_models = [m for m in CONFIG["models_to_run"] if m != proposed_model]

    for dataset_name in sorted(set(k[1] for k in keys)):
        for building in sorted(set(k[2] for k in keys if k[1] == dataset_name)):
            prop_key = (proposed_model, dataset_name, building)
            if prop_key not in predictions:
                continue
            prop_df = predictions[prop_key]
            # Error distribution for proposed.
            e_prop_all = prop_df["Actual"].values - prop_df["Prediction"].values
            for value in np.abs(e_prop_all):
                if np.isfinite(value):
                    error_rows.append({"Model": proposed_model, "Dataset": dataset_name, "Building": building, "Abs_Error": float(value)})

            for baseline in baseline_models:
                base_key = (baseline, dataset_name, building)
                if base_key not in predictions:
                    continue
                base_df = predictions[base_key]
                y, e_prop, e_base = aligned_errors(prop_df, base_df)
                if len(y) == 0:
                    continue

                # Error distribution for baseline.
                for value in np.abs(e_base):
                    if np.isfinite(value):
                        error_rows.append({"Model": baseline, "Dataset": dataset_name, "Building": building, "Abs_Error": float(value)})

                for loss in CONFIG["statistics"]["dm_loss"]:
                    dm = dm_test_errors(e_prop, e_base, loss=loss, h=CONFIG["statistics"]["dm_horizon"])
                    dm_rows.append({
                        "Dataset": dataset_name,
                        "Building": building,
                        "Proposed_Model": proposed_model,
                        "Baseline_Model": baseline,
                        "Loss": loss,
                        **dm,
                        "Proposed_Better": bool(dm["Mean_Loss_Diff_BaselineMinusProposed"] > 0) if np.isfinite(dm["Mean_Loss_Diff_BaselineMinusProposed"]) else None,
                        "Significant_at_alpha": bool(dm["P_Value"] < alpha) if np.isfinite(dm["P_Value"]) else None,
                    })

                abs_prop = np.abs(e_prop)
                abs_base = np.abs(e_base)
                finite_mask = np.isfinite(abs_prop) & np.isfinite(abs_base)
                abs_prop = abs_prop[finite_mask]
                abs_base = abs_base[finite_mask]
                try:
                    # H1: baseline absolute error > proposed absolute error.
                    stat, p_value = wilcoxon(abs_base, abs_prop, alternative="greater", zero_method="wilcox")
                    median_diff = float(np.median(abs_base - abs_prop))
                except Exception:
                    stat, p_value, median_diff = np.nan, np.nan, np.nan
                wilcoxon_rows.append({
                    "Dataset": dataset_name,
                    "Building": building,
                    "Proposed_Model": proposed_model,
                    "Baseline_Model": baseline,
                    "Alternative": "baseline_abs_error > proposed_abs_error",
                    "Statistic": float(stat) if np.isfinite(stat) else np.nan,
                    "P_Value": float(p_value) if np.isfinite(p_value) else np.nan,
                    "Median_AbsError_Diff_BaselineMinusProposed": median_diff,
                    "N": int(len(abs_prop)),
                    "Proposed_Better": bool(median_diff > 0) if np.isfinite(median_diff) else None,
                    "Significant_at_alpha": bool(p_value < alpha) if np.isfinite(p_value) else None,
                })

    return pd.DataFrame(dm_rows), pd.DataFrame(wilcoxon_rows), pd.DataFrame(error_rows)


def nemenyi_posthoc_from_metric(metric_pivot: pd.DataFrame, metric: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Nemenyi post-hoc p-value approximation from average ranks.
    Lower is better for error metrics. Higher is better for R2.
    """
    ascending = metric != "R2"
    ranks = metric_pivot.rank(axis=1, method="average", ascending=ascending)
    avg_ranks = ranks.mean(axis=0).sort_values()
    models = list(avg_ranks.index)
    k = len(models)
    n = len(ranks)
    se = np.sqrt(k * (k + 1) / (6.0 * n)) if n > 0 else np.nan

    p_matrix = pd.DataFrame(np.nan, index=models, columns=models)
    for i, mi in enumerate(models):
        for j, mj in enumerate(models):
            if i == j:
                p_matrix.loc[mi, mj] = 1.0
            else:
                diff = abs(avg_ranks[mi] - avg_ranks[mj])
                if studentized_range is not None and np.isfinite(se) and se > 0:
                    q_stat = diff / se
                    p = float(studentized_range.sf(q_stat * np.sqrt(2.0), k, np.inf))
                else:
                    p = np.nan
                p_matrix.loc[mi, mj] = p

    avg_rank_df = avg_ranks.reset_index()
    avg_rank_df.columns = ["Model", "Average_Rank"]
    avg_rank_df["Metric"] = metric
    avg_rank_df["N_Blocks"] = n
    avg_rank_df["N_Models"] = k
    return p_matrix, avg_rank_df


def run_block_level_tests(metrics_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    proposed_model = CONFIG["statistics"]["proposed_model"]
    alpha = CONFIG["statistics"]["alpha"]
    friedman_rows = []
    block_wilcoxon_rows = []
    nemenyi_tables: Dict[str, pd.DataFrame] = {}
    avg_rank_tables: Dict[str, pd.DataFrame] = {}

    df = metrics_df.copy()
    df["Block"] = df["Dataset"].astype(str) + "::" + df["Building"].astype(str)

    baseline_models = [m for m in CONFIG["models_to_run"] if m != proposed_model]
    for metric in CONFIG["statistics"]["metrics_for_block_tests"]:
        pivot = df.pivot_table(index="Block", columns="Model", values=metric, aggfunc="mean")
        pivot = pivot.dropna(axis=0, how="any")
        models = [m for m in CONFIG["models_to_run"] if m in pivot.columns]
        pivot = pivot[models]
        if len(pivot) < 2 or len(models) < 3:
            friedman_rows.append({
                "Metric": metric,
                "Friedman_Statistic": np.nan,
                "P_Value": np.nan,
                "N_Blocks": len(pivot),
                "N_Models": len(models),
                "Significant_at_alpha": None,
                "Note": "Not enough complete blocks/models for Friedman test.",
            })
            continue

        try:
            stat, p_value = friedmanchisquare(*[pivot[m].values for m in models])
        except Exception:
            stat, p_value = np.nan, np.nan
        friedman_rows.append({
            "Metric": metric,
            "Friedman_Statistic": float(stat) if np.isfinite(stat) else np.nan,
            "P_Value": float(p_value) if np.isfinite(p_value) else np.nan,
            "N_Blocks": int(len(pivot)),
            "N_Models": int(len(models)),
            "Significant_at_alpha": bool(p_value < alpha) if np.isfinite(p_value) else None,
            "Note": "Lower rank is better for this metric." if metric != "R2" else "Higher value is better for R2.",
        })

        nemenyi_p, avg_ranks = nemenyi_posthoc_from_metric(pivot, metric)
        nemenyi_tables[metric] = nemenyi_p
        avg_rank_tables[metric] = avg_ranks

        for baseline in baseline_models:
            if baseline not in pivot.columns or proposed_model not in pivot.columns:
                continue
            prop_values = pivot[proposed_model].values
            base_values = pivot[baseline].values
            if metric == "R2":
                x, y = prop_values, base_values
                alternative = "proposed_metric > baseline_metric"
                diff = prop_values - base_values
            else:
                x, y = base_values, prop_values
                alternative = "baseline_error_metric > proposed_error_metric"
                diff = base_values - prop_values
            try:
                stat_w, p_w = wilcoxon(x, y, alternative="greater", zero_method="wilcox")
                median_diff = float(np.median(diff))
            except Exception:
                stat_w, p_w, median_diff = np.nan, np.nan, np.nan
            block_wilcoxon_rows.append({
                "Metric": metric,
                "Proposed_Model": proposed_model,
                "Baseline_Model": baseline,
                "Alternative": alternative,
                "Statistic": float(stat_w) if np.isfinite(stat_w) else np.nan,
                "P_Value": float(p_w) if np.isfinite(p_w) else np.nan,
                "Median_Diff_BaselineMinusProposed": median_diff,
                "N_Blocks": int(len(pivot)),
                "Proposed_Better": bool(median_diff > 0) if np.isfinite(median_diff) else None,
                "Significant_at_alpha": bool(p_w < alpha) if np.isfinite(p_w) else None,
            })

    return pd.DataFrame(friedman_rows), pd.DataFrame(block_wilcoxon_rows), nemenyi_tables, avg_rank_tables


def create_error_plots(error_df: pd.DataFrame) -> None:
    if error_df.empty:
        return
    max_points = CONFIG["statistics"]["max_plot_points_per_model"]
    plot_df_parts = []
    for model_name, g in error_df.groupby("Model"):
        if len(g) > max_points:
            plot_df_parts.append(g.sample(max_points, random_state=CONFIG["seed"]))
        else:
            plot_df_parts.append(g)
    plot_df = pd.concat(plot_df_parts, ignore_index=True)
    models = [m for m in CONFIG["models_to_run"] if m in plot_df["Model"].unique()]
    data = [plot_df.loc[plot_df["Model"] == m, "Abs_Error"].dropna().values for m in models]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=models, showfliers=False)
    plt.ylabel("Absolute Error")
    plt.xlabel("Model")
    plt.title("Absolute Error Distribution: Proposed Model vs Baselines")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    box_path = os.path.join(DIRS["plots"], "absolute_error_boxplot_all.png")
    plt.savefig(box_path, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.violinplot(data, showmeans=True, showmedians=True)
    plt.xticks(np.arange(1, len(models) + 1), models, rotation=20, ha="right")
    plt.ylabel("Absolute Error")
    plt.xlabel("Model")
    plt.title("Absolute Error Distribution: Violin Plot")
    plt.tight_layout()
    violin_path = os.path.join(DIRS["plots"], "absolute_error_violin_all.png")
    plt.savefig(violin_path, dpi=300)
    plt.close()

    print(f"  Saved box plot   : {box_path}")
    print(f"  Saved violin plot: {violin_path}")


def run_all_statistics(predictions: Dict[Tuple[str, str, str], pd.DataFrame], metrics_df: pd.DataFrame) -> None:
    print(f"\n[{now_str()}] Running statistical confirmation tests")
    dm_df, wilcoxon_error_df, error_df = run_error_sequence_tests(predictions)
    friedman_df, block_wilcoxon_df, nemenyi_tables, avg_rank_tables = run_block_level_tests(metrics_df)

    dm_path = os.path.join(DIRS["statistics"], "dm_tests.csv")
    wilcox_error_path = os.path.join(DIRS["statistics"], "wilcoxon_error_tests.csv")
    friedman_path = os.path.join(DIRS["statistics"], "friedman_tests.csv")
    block_wilcox_path = os.path.join(DIRS["statistics"], "block_level_wilcoxon_tests.csv")
    error_path = os.path.join(DIRS["statistics"], "absolute_error_long.csv")

    dm_df.to_csv(dm_path, index=False)
    wilcoxon_error_df.to_csv(wilcox_error_path, index=False)
    friedman_df.to_csv(friedman_path, index=False)
    block_wilcoxon_df.to_csv(block_wilcox_path, index=False)
    error_df.to_csv(error_path, index=False)

    for metric, table in nemenyi_tables.items():
        table.to_csv(os.path.join(DIRS["statistics"], f"nemenyi_{metric}.csv"))
    for metric, table in avg_rank_tables.items():
        table.to_csv(os.path.join(DIRS["statistics"], f"average_ranks_{metric}.csv"), index=False)

    create_error_plots(error_df)

    print(f"  Saved DM tests                  : {dm_path}")
    print(f"  Saved Wilcoxon error tests      : {wilcox_error_path}")
    print(f"  Saved Friedman tests            : {friedman_path}")
    print(f"  Saved block-level Wilcoxon tests: {block_wilcox_path}")


# ============================================================
# 10. MAIN
# ============================================================
def run_experiment_for_seed(seed: int, df_all: pd.DataFrame, df_main: pd.DataFrame,
                            df_holdout: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[Tuple[str, str, str], pd.DataFrame]]:
    """Run every requested model once for a single seed. Returns metrics rows
    (each tagged with Seed), training history, and the predictions dict."""
    CONFIG["seed"] = seed          # downstream reseeds read CONFIG["seed"]
    set_seed(seed)
    print("\n" + "#" * 80)
    print(f"# SEED {seed}")
    print("#" * 80)

    seed_metrics: List[Dict[str, Any]] = []
    seed_history: List[Dict[str, Any]] = []
    seed_predictions: Dict[Tuple[str, str, str], pd.DataFrame] = {}

    # Determine which models to run (including ablation variants).
    raw_baselines = [m for m in CONFIG["models_to_run"] if m in ["GRU", "LSTM", "Transformer"]]
    if CONFIG.get("run_ablation", False):
        raw_baselines = raw_baselines + ["TCN"]    # raw-signal TCN ablation

    # Baseline + raw-TCN models share the same raw-data split and scaler.
    if raw_baselines:
        raw_bundle = prepare_raw_bundle(df_main)
        for model_name in raw_baselines:
            metrics_rows, history_rows, predictions = run_baseline_model(model_name, raw_bundle, df_all, df_main, df_holdout)
            seed_metrics.extend(metrics_rows)
            seed_history.extend(history_rows)
            seed_predictions.update(predictions)

    # VMD-based model (main proposed model). Only one VMD+TCN configuration is
    # trained now (fixed LR, no Optuna); the raw-signal "TCN" ablation above is
    # what isolates the VMD decomposition's contribution.
    if "VMD_TCN" in CONFIG["models_to_run"]:
        vmd_bundle = prepare_vmd_bundle(df_main)
        metrics_rows, history_rows, predictions = run_vmd_tcn(
            vmd_bundle, df_all, df_main, df_holdout, model_name="VMD_TCN")
        seed_metrics.extend(metrics_rows)
        seed_history.extend(history_rows)
        seed_predictions.update(predictions)

    for row in seed_metrics:
        row["Seed"] = seed
    for row in seed_history:
        row["Seed"] = seed
    return seed_metrics, seed_history, seed_predictions


def aggregate_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std of each metric across seeds, per Model/Dataset/Building."""
    metric_cols = [c for c in ["MAE", "MSE", "RMSE", "MAPE", "SMAPE", "R2",
                               "Training_Time_sec", "Testing_Time_sec"] if c in metrics_df.columns]
    grouped = metrics_df.groupby(["Model", "Dataset", "Building"])[metric_cols]
    agg = grouped.agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg.reset_index()


def main() -> None:
    ensure_dirs()
    setup_plot_fonts()
    total_start = time.time()
    print("=" * 80)
    print("ONE-FILE EXPERIMENT: VMD_TCN vs GRU, LSTM, Transformer")
    print("=" * 80)
    print(f"Start time     : {now_str()}")
    print(f"Device         : {CONFIG['device']}")
    print(f"Seeds          : {CONFIG.get('seeds', [CONFIG['seed']])}")
    print(f"Run ablation   : {CONFIG.get('run_ablation', False)}")
    print(f"Tune baselines : {CONFIG.get('tune_baselines', False)}")
    print(f"VMD scaler fit : {CONFIG['vmd']['scale_fit_target']}  | eval target: {CONFIG['vmd']['eval_target']}  | decompose: {CONFIG['vmd'].get('decompose_mode','global')}")
    print(f"Output dir     : {OUTPUT_DIR}")

    save_json(os.path.join(OUTPUT_DIR, "config_used.json"), CONFIG)

    df_all, df_main, df_holdout = load_data()

    seeds = CONFIG.get("seeds", [CONFIG["seed"]])
    all_metrics: List[Dict[str, Any]] = []
    all_history: List[Dict[str, Any]] = []
    stats_predictions: Dict[Tuple[str, str, str], pd.DataFrame] = {}

    for i, seed in enumerate(seeds):
        seed_metrics, seed_history, seed_predictions = run_experiment_for_seed(seed, df_all, df_main, df_holdout)
        all_metrics.extend(seed_metrics)
        all_history.extend(seed_history)
        # Keep the FIRST seed's predictions for the sequence-level statistics and
        # for the per-building prediction CSVs the notebook plots (prediction keys
        # are (model, dataset, building) and would otherwise collide across seeds).
        if i == 0:
            stats_predictions = seed_predictions

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(DIRS["metrics"], "all_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # Aggregated mean +/- std across seeds (the table the paper should report).
    if len(seeds) > 1:
        agg_df = aggregate_metrics(metrics_df)
        agg_path = os.path.join(DIRS["metrics"], "metrics_aggregated.csv")
        agg_df.to_csv(agg_path, index=False)
        print(f"Saved aggregated metrics (mean+/-std over {len(seeds)} seeds): {agg_path}")

    history_df = pd.DataFrame(all_history)
    history_df.to_csv(os.path.join(DIRS["metrics"], "training_history.csv"), index=False)

    # Model complexity (params + train time) for the computational-cost table.
    if COMPLEXITY_ROWS:
        complexity_path = os.path.join(DIRS["metrics"], "model_complexity.csv")
        pd.DataFrame(COMPLEXITY_ROWS).to_csv(complexity_path, index=False)
        print(f"Saved model complexity: {complexity_path}")

    if CONFIG["save_long_prediction_file"]:
        long_predictions = pd.concat(stats_predictions.values(), ignore_index=True) if stats_predictions else pd.DataFrame()
        long_pred_path = os.path.join(DIRS["predictions"], "all_predictions_long.csv")
        long_predictions.to_csv(long_pred_path, index=False)
        print(f"Saved long predictions: {long_pred_path}")

    print(f"Saved metrics         : {metrics_path}")

    # Statistics run on the first seed (a single, coherent set of error series).
    first_seed_metrics = metrics_df[metrics_df.get("Seed", seeds[0]) == seeds[0]] if "Seed" in metrics_df.columns else metrics_df
    run_all_statistics(stats_predictions, first_seed_metrics)

    total_time = time.time() - total_start
    print("=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"End time  : {now_str()}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Check output folder: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
