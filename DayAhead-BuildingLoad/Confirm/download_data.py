"""
Data download and preparation only for the electricity forecasting experiment.

This script does NOT train any model.
It only:
  1. Downloads the Building Data Genome Project 2 dataset by KaggleHub.
  2. Finds electricity_cleaned.csv.
  3. Copies it to:
       BASE_DIR/meters/cleaned/electricity_cleaned.csv
  4. Validates the timestamp column and required building columns.
  5. Saves small preparation reports for reproducibility.

Required package:
  pip install kagglehub pandas numpy

Run:
  python download_data_prepare_only.py

Optional:
  python download_data_prepare_only.py --force
  python download_data_prepare_only.py --no-validation
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "meters", "cleaned")
ELECTRICITY_FILE = os.path.join(DATA_DIR, "electricity_cleaned.csv")

REPORT_DIR = os.path.join(BASE_DIR, "data_preparation_report")

KAGGLE_DATASET = "claytonmiller/buildingdatagenomeproject2"
SOURCE_FILENAME = "electricity_cleaned.csv"


# ============================================================
# 2. CONFIGURATION FOR CHECKING ONLY
#    These names should match the experiment script.
# ============================================================
CONFIG = {
    "target_buildings": ["Wolf", "Bull", "Robin", "Fox", "Rat", "Eagle"],
    "test_buildings": ["Wolf", "Bull", "Robin", "Fox", "Rat", "Eagle", "Hog"],
    "holdout_start": "2017-12-01 00:00:00",
    "holdout_end": "2017-12-31 23:00:00",
    "split_ratio": {"train": 0.7, "val": 0.1, "test": 0.2},
}


# ============================================================
# 3. UTILITIES
# ============================================================
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


def copy_large_file(src_path: str, dst_path: str, chunk_size: int = 1024 * 1024 * 32) -> None:
    """Copy a large file in chunks."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
        shutil.copyfileobj(src, dst, length=chunk_size)


def find_file_by_name(root_dir: str, filename: str) -> Optional[str]:
    """Find the first file matching filename inside root_dir."""
    target = filename.lower()
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower() == target:
                return os.path.join(root, file)
    return None


def list_csv_preview(root_dir: str, max_files: int = 50) -> List[str]:
    """Return a preview list of CSV files found under root_dir."""
    found = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                found.append(os.path.relpath(os.path.join(root, file), root_dir))
                if len(found) >= max_files:
                    return found
    return found


def find_building_column(df: pd.DataFrame, building: str) -> Optional[str]:
    """Find the first column containing a building name, case-insensitive."""
    matches = [c for c in df.columns if building.lower() in c.lower()]
    return matches[0] if matches else None


def write_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


# ============================================================
# 4. DOWNLOAD AND COPY
# ============================================================
def download_and_copy_electricity_file(force: bool = False) -> str:
    """
    Download the KaggleHub dataset and copy electricity_cleaned.csv
    to BASE_DIR/meters/cleaned/electricity_cleaned.csv.
    """
    ensure_dirs()

    if os.path.exists(ELECTRICITY_FILE) and not force:
        print(f"[{now_str()}] Data already exists:")
        print(f"  {ELECTRICITY_FILE}")
        print("Use --force to re-download and overwrite it.")
        return ELECTRICITY_FILE

    try:
        import kagglehub
    except Exception as exc:
        raise ImportError(
            "kagglehub is required to download the dataset.\n"
            "Install it first:\n"
            "  pip install kagglehub\n\n"
            f"Original error: {exc}"
        )

    print("=" * 80)
    print("DATA DOWNLOAD")
    print("=" * 80)
    print(f"[{now_str()}] Downloading dataset:")
    print(f"  {KAGGLE_DATASET}")

    downloaded_path = kagglehub.dataset_download(KAGGLE_DATASET)

    print(f"[{now_str()}] KaggleHub downloaded path:")
    print(f"  {downloaded_path}")

    source_csv = find_file_by_name(downloaded_path, SOURCE_FILENAME)

    if source_csv is None:
        csv_preview = list_csv_preview(downloaded_path)
        preview_text = "\n".join(csv_preview) if csv_preview else "No CSV files found."
        raise FileNotFoundError(
            f"Could not find {SOURCE_FILENAME} inside the downloaded dataset.\n"
            f"Downloaded path: {downloaded_path}\n\n"
            f"CSV preview:\n{preview_text}"
        )

    print(f"[{now_str()}] Found source file:")
    print(f"  {source_csv}")

    print(f"[{now_str()}] Copying to experiment data path:")
    print(f"  {ELECTRICITY_FILE}")

    copy_large_file(source_csv, ELECTRICITY_FILE)

    size_mb = os.path.getsize(ELECTRICITY_FILE) / (1024 * 1024)
    print(f"[{now_str()}] Copy complete.")
    print(f"  Size: {size_mb:.2f} MB")

    return ELECTRICITY_FILE


# ============================================================
# 5. VALIDATION AND PREPARATION REPORTS
# ============================================================
def validate_and_report(csv_path: str) -> None:
    """
    Validate electricity_cleaned.csv and save reproducibility reports.
    This does not create model inputs and does not run training.
    """
    ensure_dirs()

    print("=" * 80)
    print("DATA VALIDATION AND REPORT")
    print("=" * 80)
    print(f"[{now_str()}] Reading CSV:")
    print(f"  {csv_path}")

    df = pd.read_csv(csv_path)

    if "timestamp" not in df.columns:
        raise ValueError(
            "The file does not contain a 'timestamp' column. "
            "The experiment script requires this column."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    invalid_ts = int(df["timestamp"].isna().sum())
    if invalid_ts > 0:
        raise ValueError(
            f"The timestamp column contains {invalid_ts} invalid values. "
            "Please check the source CSV."
        )

    # Sort by timestamp only if needed.
    was_sorted = bool(df["timestamp"].is_monotonic_increasing)
    if not was_sorted:
        print(f"[{now_str()}] Timestamp is not sorted. Sorting and overwriting prepared CSV.")
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.to_csv(csv_path, index=False)

    selected_rows = []
    missing_buildings = []

    for building in CONFIG["test_buildings"]:
        col = find_building_column(df, building)
        if col is None:
            missing_buildings.append(building)
            selected_rows.append({
                "Building": building,
                "Column": "",
                "Found": False,
                "Missing_Values": np.nan,
                "Missing_Percent": np.nan,
                "Min": np.nan,
                "Max": np.nan,
                "Mean": np.nan,
                "Std": np.nan,
            })
        else:
            series = pd.to_numeric(df[col], errors="coerce")
            missing = int(series.isna().sum())
            selected_rows.append({
                "Building": building,
                "Column": col,
                "Found": True,
                "Missing_Values": missing,
                "Missing_Percent": float(missing / max(1, len(series)) * 100.0),
                "Min": float(series.min()) if series.notna().any() else np.nan,
                "Max": float(series.max()) if series.notna().any() else np.nan,
                "Mean": float(series.mean()) if series.notna().any() else np.nan,
                "Std": float(series.std()) if series.notna().any() else np.nan,
            })

    columns_df = pd.DataFrame(selected_rows)
    columns_report_path = os.path.join(REPORT_DIR, "selected_building_columns.csv")
    columns_df.to_csv(columns_report_path, index=False)

    holdout_start = pd.to_datetime(CONFIG["holdout_start"])
    holdout_end = pd.to_datetime(CONFIG["holdout_end"])
    holdout_mask = (df["timestamp"] >= holdout_start) & (df["timestamp"] <= holdout_end)
    df_holdout = df[holdout_mask]
    df_main = df[~holdout_mask]

    train_len = int(len(df_main) * CONFIG["split_ratio"]["train"])
    val_len = int(len(df_main) * CONFIG["split_ratio"]["val"])
    test_len = len(df_main) - train_len - val_len

    split_report = {
        "main_rows_excluding_holdout": int(len(df_main)),
        "holdout_rows": int(len(df_holdout)),
        "train_rows_estimated": int(train_len),
        "val_rows_estimated": int(val_len),
        "test_rows_estimated": int(test_len),
        "split_ratio": CONFIG["split_ratio"],
        "holdout_start": CONFIG["holdout_start"],
        "holdout_end": CONFIG["holdout_end"],
    }

    summary = {
        "prepared_at": now_str(),
        "base_dir": BASE_DIR,
        "data_dir": DATA_DIR,
        "electricity_file": ELECTRICITY_FILE,
        "source_dataset": KAGGLE_DATASET,
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "timestamp_min": str(df["timestamp"].min()),
        "timestamp_max": str(df["timestamp"].max()),
        "timestamp_was_sorted": was_sorted,
        "missing_target_or_test_buildings": missing_buildings,
        "target_buildings": CONFIG["target_buildings"],
        "test_buildings": CONFIG["test_buildings"],
        **split_report,
    }

    summary_path = os.path.join(REPORT_DIR, "data_preparation_summary.json")
    write_json(summary_path, summary)

    all_columns_path = os.path.join(REPORT_DIR, "all_columns.txt")
    with open(all_columns_path, "w", encoding="utf-8") as f:
        for col in df.columns:
            f.write(str(col) + "\n")

    print(f"[{now_str()}] Validation complete.")
    print(f"  Rows             : {len(df)}")
    print(f"  Columns          : {len(df.columns)}")
    print(f"  Timestamp range  : {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"  Data file        : {ELECTRICITY_FILE}")
    print(f"  Summary report   : {summary_path}")
    print(f"  Building columns : {columns_report_path}")
    print(f"  All columns      : {all_columns_path}")

    if missing_buildings:
        print("\nWARNING: These buildings were not found:")
        for building in missing_buildings:
            print(f"  - {building}")
        print("The experiment script will skip missing buildings or fail if required columns are absent.")


# ============================================================
# 6. MAIN
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare electricity_cleaned.csv only. No model training is included."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite meters/cleaned/electricity_cleaned.csv if it already exists.",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Only download/copy the CSV. Skip validation and report generation.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("ELECTRICITY DATA DOWNLOAD AND PREPARATION ONLY")
    print("=" * 80)
    print(f"Base dir        : {BASE_DIR}")
    print(f"Data dir        : {DATA_DIR}")
    print(f"Target CSV path : {ELECTRICITY_FILE}")
    print("Model training  : DISABLED")

    try:
        csv_path = download_and_copy_electricity_file(force=args.force)

        if not args.no_validation:
            validate_and_report(csv_path)

        print("=" * 80)
        print("DATA PREPARATION COMPLETE")
        print("=" * 80)
        print("Now you can run your experiment script, which should read:")
        print(f"  {ELECTRICITY_FILE}")

    except Exception as exc:
        print("=" * 80)
        print("DATA PREPARATION FAILED")
        print("=" * 80)
        print(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
