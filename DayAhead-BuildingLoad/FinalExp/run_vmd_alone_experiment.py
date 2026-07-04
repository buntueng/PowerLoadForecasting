"""
Extra experiment: VMD alone (no TCN, no forecasting), swept over K and alpha.

IMPORTANT — what this measures:
  VMD is a signal-DECOMPOSITION method, not a forecaster, so "testing VMD alone"
  cannot mean forecasting accuracy (there is no model making predictions).
  What this script reports instead is VMD's RECONSTRUCTION FIDELITY: for each
  (K, alpha) setting, the signal is decomposed into K modes, the modes are
  summed back together, and that reconstruction is compared against the true
  cleaned signal with MAE / RMSE / MAPE / SMAPE / R2. This tells you how much
  information VMD loses/distorts at each K, alpha — the standard way to justify
  a K/alpha choice before it is fed into the TCN. If you actually meant
  "VMD + TCN with no Optuna" for varying K/alpha, that is a different
  experiment (extend run_vmd_optuna_tcn with use_optuna=False per K/alpha);
  ask and I'll add it.

7 cases (as requested):
  K=4, alpha=2000
  K=5, alpha=2000
  K=6, alpha=2000
  K=7, alpha=2000
  K=8, alpha=2000
  K=6, alpha=1000
  K=6, alpha=3000

For each case, reconstruction fidelity is computed separately on:
  - "test"    : the held-out test slice of each of the 6 training buildings
                (same split_ratio as the rest of the pipeline)
  - "holdout" : the Dec-2017 month for those same buildings
  - "cross"   : the unseen "Hog" building
per building, plus a "MEAN" aggregate row per (K, alpha, Dataset).

This reuses clean_flat_sequences / find_building_column / compute_metrics from
run_all_experiments.py so numbers are consistent with the rest of your results.
It does NOT touch results_all_experiments/ — outputs go to results_vmd_alone/.

Requirements: numpy, pandas, scikit-learn, scipy, vmdpy.
Place this file next to run_all_experiments.py and run:

    python run_vmd_alone_experiment.py

Output:
  results_vmd_alone/vmd_alone_metrics_per_building.csv   (every building x case x dataset row)
  results_vmd_alone/vmd_alone_metrics_summary.csv        (mean across buildings per case x dataset)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_experiments as base  # noqa: E402
import pandas as pd  # noqa: E402

if base.VMD is None:
    raise ImportError(
        f"vmdpy is required for this script. Install with: pip install vmdpy. "
        f"Original error: {base.VMD_IMPORT_ERROR}"
    )

# The 7 requested (K, alpha) cases.
CASES = [
    {"K": 4, "alpha": 2000},
    {"K": 5, "alpha": 2000},
    {"K": 6, "alpha": 2000},
    {"K": 7, "alpha": 2000},
    {"K": 8, "alpha": 2000},
    {"K": 6, "alpha": 1000},
    {"K": 6, "alpha": 3000},
]
TAU, DC, INIT, TOL = 0, 0, 1, 1e-7


def decompose_and_score(series, K, alpha, eval_slice=None):
    """Run VMD, sum the modes, and score the reconstruction against the true
    series (optionally restricted to eval_slice, a slice object)."""
    modes, _, _ = base.VMD(series, alpha, TAU, K, DC, INIT, TOL)
    reconstruction = modes.sum(axis=0)
    if eval_slice is not None:
        actual = series[eval_slice]
        recon = reconstruction[eval_slice]
    else:
        actual = series
        recon = reconstruction
    n = min(len(actual), len(recon))
    metrics = base.compute_metrics(actual[:n], recon[:n])
    metrics["N_Points"] = n
    return metrics


def main() -> None:
    out_dir = os.path.join(base.BASE_DIR, "results_vmd_alone")
    os.makedirs(out_dir, exist_ok=True)

    split_ratio = base.CONFIG["split_ratio"]
    target_buildings = base.CONFIG["target_buildings"]
    extra_test_buildings = [b for b in base.CONFIG["test_buildings"] if b not in target_buildings]

    df_all, df_main, df_holdout = base.load_data()

    rows = []
    for case in CASES:
        K, alpha = case["K"], case["alpha"]
        print(f"\n{'=' * 80}\nCase: K={K}, alpha={alpha}\n{'=' * 80}")

        # --- "test" dataset: held-out tail of each target building ---
        for bid in target_buildings:
            col = base.find_building_column(df_main, bid)
            if col is None:
                print(f"  Skip {bid}: column not found")
                continue
            series = base.clean_flat_sequences(df_main[col].values)
            total_len = len(series)
            train_len = int(total_len * split_ratio["train"])
            val_len = int(total_len * split_ratio["val"])
            test_slice = slice(train_len + val_len, None)
            print(f"  [test]    {bid}: VMD K={K} alpha={alpha} on {total_len} samples")
            m = decompose_and_score(series, K, alpha, eval_slice=test_slice)
            rows.append({"K": K, "Alpha": alpha, "Dataset": "test", "Building": bid, **m})

        # --- "holdout" dataset: Dec-2017 month for the same buildings ---
        if len(df_holdout) > 0:
            for bid in target_buildings:
                col = base.find_building_column(df_holdout, bid)
                if col is None:
                    continue
                series = base.clean_flat_sequences(df_holdout[col].values)
                print(f"  [holdout] {bid}: VMD K={K} alpha={alpha} on {len(series)} samples")
                m = decompose_and_score(series, K, alpha, eval_slice=None)
                rows.append({"K": K, "Alpha": alpha, "Dataset": "holdout", "Building": bid, **m})

        # --- "cross" dataset: unseen building(s), e.g. Hog ---
        for test_bid in extra_test_buildings:
            col = base.find_building_column(df_all, test_bid)
            if col is None:
                print(f"  Skip cross-building {test_bid}: column not found")
                continue
            mask = df_all["timestamp"] >= pd.to_datetime(base.CONFIG["hog_test_start"])
            df_cross = df_all[mask].reset_index(drop=True)
            series = base.clean_flat_sequences(df_cross[col].values)
            print(f"  [cross]   {test_bid}: VMD K={K} alpha={alpha} on {len(series)} samples")
            m = decompose_and_score(series, K, alpha, eval_slice=None)
            rows.append({"K": K, "Alpha": alpha, "Dataset": "cross", "Building": test_bid, **m})

    per_building_df = pd.DataFrame(rows)
    per_building_path = os.path.join(out_dir, "vmd_alone_metrics_per_building.csv")
    per_building_df.to_csv(per_building_path, index=False)

    # Mean across buildings for each (K, alpha, Dataset) — the headline table.
    metric_cols = ["MAE", "MSE", "RMSE", "MAPE", "SMAPE", "R2"]
    summary_df = (
        per_building_df.groupby(["K", "Alpha", "Dataset"])[metric_cols]
        .mean()
        .reset_index()
        .sort_values(["Dataset", "K", "Alpha"])
    )
    summary_path = os.path.join(out_dir, "vmd_alone_metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("VMD-ALONE SWEEP COMPLETE")
    print(f"Per-building metrics : {per_building_path}")
    print(f"Summary (mean/case)  : {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
