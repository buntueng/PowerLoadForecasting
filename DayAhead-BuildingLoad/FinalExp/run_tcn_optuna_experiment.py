"""
Extra experiment: TCN + Optuna (raw signal, no VMD).

This trains the SAME TCN architecture used inside VMD_Optuna_TCN, but directly
on the raw (un-decomposed) signal, with its learning rate tuned by Optuna using
the same search budget as the proposed model (CONFIG["optuna"]). This isolates
"how much does Optuna-tuning alone buy a plain TCN" as a companion result to
the "TCN" ablation (which uses a fixed LR) and to "VMD_Optuna_TCN".

It reuses every function from run_all_experiments.py (data cleaning, dataset
building, the TCN model, training loop, Optuna tuner, metric computation) so
results stay perfectly consistent with the rest of your pipeline. It does NOT
touch or overwrite results_all_experiments/ — outputs go to a separate
results_tcn_optuna/ folder.

Requirements: same as run_all_experiments.py (numpy, pandas, scikit-learn,
scipy, matplotlib, torch, optuna). vmdpy is not needed for this script.

Place this file in the SAME directory as run_all_experiments.py (so the data
path meters/cleaned/electricity_cleaned.csv resolves correctly), then run:

    python run_tcn_optuna_experiment.py

Output:
  results_tcn_optuna/metrics/tcn_optuna_metrics.csv            (full per-row metrics)
  results_tcn_optuna/metrics/tcn_optuna_metrics_summary.csv    (Model/Dataset/Building/Seed + MAE/RMSE/MAPE/SMAPE/R2 only)
  results_tcn_optuna/metrics/tcn_optuna_metrics_aggregated.csv (mean+/-std across seeds, only if len(seeds) > 1)
  results_tcn_optuna/metrics/optuna_TCN_trials.csv             (Optuna trial history)
  results_tcn_optuna/predictions/TCN/...                        (per-building prediction CSVs)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_experiments as base  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> None:
    # --- Point every output path at a dedicated folder, isolated from
    # results_all_experiments/ so nothing from a previous full run is touched. ---
    out_dir = os.path.join(base.BASE_DIR, "results_tcn_optuna")
    base.OUTPUT_DIR = out_dir
    base.DIRS = {
        "models": os.path.join(out_dir, "models"),
        "predictions": os.path.join(out_dir, "predictions"),
        "metrics": os.path.join(out_dir, "metrics"),
        "statistics": os.path.join(out_dir, "statistics"),
        "plots": os.path.join(out_dir, "statistics", "plots"),
    }
    for p in base.DIRS.values():
        os.makedirs(p, exist_ok=True)
    os.makedirs(os.path.join(base.DIRS["models"], "TCN"), exist_ok=True)
    os.makedirs(os.path.join(base.DIRS["predictions"], "TCN"), exist_ok=True)

    base.setup_plot_fonts()
    base.COMPLEXITY_ROWS.clear()

    # --- Force this run to be exactly "TCN, Optuna-tuned, raw signal" ---
    base.CONFIG["tune_baselines"] = True         # turn Optuna LR search ON for this baseline
    base.CONFIG["reuse_existing_checkpoints"] = False

    seeds = base.CONFIG.get("seeds", [base.CONFIG["seed"]])
    print(f"Running TCN + Optuna over seeds: {seeds}")

    df_all, df_main, df_holdout = base.load_data()

    all_metrics = []
    all_history = []
    for seed in seeds:
        base.CONFIG["seed"] = seed
        base.set_seed(seed)
        print("\n" + "#" * 80)
        print(f"# TCN + Optuna | SEED {seed}")
        print("#" * 80)

        raw_bundle = base.prepare_raw_bundle(df_main)
        metrics_rows, history_rows, _predictions = base.run_baseline_model(
            "TCN", raw_bundle, df_all, df_main, df_holdout
        )
        for row in metrics_rows:
            row["Seed"] = seed
        for row in history_rows:
            row["Seed"] = seed
        all_metrics.extend(metrics_rows)
        all_history.extend(history_rows)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(base.DIRS["metrics"], "tcn_optuna_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    summary_cols = ["Model", "Dataset", "Building", "Seed", "MAE", "RMSE", "MAPE", "SMAPE", "R2"]
    summary_df = metrics_df[[c for c in summary_cols if c in metrics_df.columns]]
    summary_path = os.path.join(base.DIRS["metrics"], "tcn_optuna_metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    if len(seeds) > 1:
        agg_df = base.aggregate_metrics(metrics_df)
        agg_path = os.path.join(base.DIRS["metrics"], "tcn_optuna_metrics_aggregated.csv")
        agg_df.to_csv(agg_path, index=False)
        print(f"Saved aggregated metrics (mean+/-std over {len(seeds)} seeds): {agg_path}")

    history_df = pd.DataFrame(all_history)
    history_df.to_csv(os.path.join(base.DIRS["metrics"], "tcn_optuna_training_history.csv"), index=False)

    if base.COMPLEXITY_ROWS:
        pd.DataFrame(base.COMPLEXITY_ROWS).to_csv(
            os.path.join(base.DIRS["metrics"], "tcn_optuna_model_complexity.csv"), index=False
        )

    print("\n" + "=" * 80)
    print("TCN + OPTUNA EXPERIMENT COMPLETE")
    print(f"Full metrics    : {metrics_path}")
    print(f"Summary metrics : {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
