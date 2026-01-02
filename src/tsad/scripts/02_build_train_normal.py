from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tsad.config import load_config, get_paths
from tsad.data.download import download_skab_repo_zip
from tsad.data.load import load_one_csv


def main() -> None:
    # -------------------------------------------------
    # Load config and resolve paths
    # -------------------------------------------------
    cfg = load_config("configs/default.yaml")
    paths = get_paths(cfg)

    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Ensure SKAB dataset exists locally
    # -------------------------------------------------
    extracted_repo_dir = download_skab_repo_zip(
        cfg["skab"]["repo_zip_url"],
        paths.raw_dir,
    )

    base = (
        extracted_repo_dir
        / cfg["skab"]["extracted_root_dirname"]
        / cfg["skab"]["data_subdir"]
    )

    # -------------------------------------------------
    # STEP 1 — Load anomaly-free normal training file
    # -------------------------------------------------
    normal_rel = "anomaly-free/anomaly-free.csv"
    normal_path = base / normal_rel

    print(f"[INFO] Loading normal training file: {normal_rel}")
    df = load_one_csv(normal_path)

    print(f"[INFO] Raw shape: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"[INFO] Columns: {list(df.columns)}")

    # -------------------------------------------------
    # STEP 2 — Parse + sort timestamp
    # -------------------------------------------------
    ts_col = cfg["dataset_prep"]["timestamp_col"]
    if ts_col not in df.columns:
        raise ValueError(f"Timestamp column '{ts_col}' not found in data")

    before_ts = len(df)
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).copy()
    after_ts = len(df)

    df = df.sort_values(ts_col).reset_index(drop=True)

    print(f"[INFO] Dropped {before_ts - after_ts} rows due to invalid timestamps")
    print(f"[INFO] Time range: {df[ts_col].min()} -> {df[ts_col].max()}")
    print(f"[INFO] Time monotonic increasing: {df[ts_col].is_monotonic_increasing}")

    # -------------------------------------------------
    # STEP 3 — Feature selection + numeric coercion + strict NaN policy
    # -------------------------------------------------
    anomaly_col = cfg["dataset_prep"]["anomaly_col"]

    drop_cols = [ts_col]
    if anomaly_col in df.columns:
        drop_cols.append(anomaly_col)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    print(f"[INFO] Feature columns ({len(feature_cols)}): {feature_cols}")

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before_feat = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    after_feat = len(df)

    print(f"[INFO] Dropped {before_feat - after_feat} rows due to missing sensor values")
    print(f"[INFO] Final rows: {after_feat}")

    # -------------------------------------------------
    # STEP 4 — Save parquet + metadata JSON
    # -------------------------------------------------
    out_parquet = paths.processed_dir / "train_normal.parquet"
    out_meta = paths.processed_dir / "train_normal_meta.json"

    df.to_parquet(out_parquet, index=False)

    meta = {
        "source_file": normal_rel,
        "rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "features": feature_cols,
        "timestamp_col": ts_col,
        "time_start": str(df[ts_col].min()),
        "time_end": str(df[ts_col].max()),
        "dtypes": {c: str(df[c].dtype) for c in [ts_col] + feature_cols},
    }

    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[SUCCESS] Saved parquet: {out_parquet}")
    print(f"[SUCCESS] Saved metadata: {out_meta}")


if __name__ == "__main__":
    main()
