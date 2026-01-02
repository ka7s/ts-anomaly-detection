from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsad.config import load_config, get_paths
from tsad.data.download import download_skab_repo_zip
from tsad.data.load import load_one_csv


def iter_strict_normal_windows(
    values_norm: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int,
) -> list[np.ndarray]:
    """
    Return windows where ALL labels in the window are 0.
    values_norm: (n_rows, n_features) already normalized with training scaler
    labels: (n_rows,) int 0/1
    """
    windows = []
    n = len(labels)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        if labels[start:end].max() == 0:
            windows.append(values_norm[start:end])
    return windows


def main() -> None:
    cfg = load_config("configs/default.yaml")
    paths = get_paths(cfg)

    manifest_path = paths.processed_dir / "skab_manifest.json"
    scaler_path = paths.processed_dir / "windowing_scaler.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler config: {scaler_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files_with_anomaly = manifest["files_with_anomaly"]

    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    feature_cols = scaler["features"]
    ts_col = scaler["timestamp_col"]
    window_size = int(scaler["window_size"])
    stride = int(scaler["stride"])
    mean = np.array(scaler["mean"], dtype=np.float64)
    std = np.array(scaler["std_safe"], dtype=np.float64)

    anomaly_col = cfg["dataset_prep"]["anomaly_col"]

    extracted_repo_dir = download_skab_repo_zip(cfg["skab"]["repo_zip_url"], paths.raw_dir)
    base = (
        extracted_repo_dir
        / cfg["skab"]["extracted_root_dirname"]
        / cfg["skab"]["data_subdir"]
    )

    all_windows: list[np.ndarray] = []
    sources: list[str] = []

    for rel in files_with_anomaly:
        csv_path = base / rel
        df = load_one_csv(csv_path)

        # clean timestamp + sort
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

        if anomaly_col not in df.columns:
            continue

        # numeric features
        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=feature_cols).reset_index(drop=True)

        labels = pd.to_numeric(df[anomaly_col], errors="coerce").fillna(0).to_numpy(dtype=int)
        values = df[feature_cols].to_numpy(dtype=np.float64)
        values_norm = (values - mean) / std

        windows = iter_strict_normal_windows(values_norm, labels, window_size, stride)
        if windows:
            all_windows.extend(windows)
            sources.extend([rel] * len(windows))

    if not all_windows:
        raise RuntimeError("No strict-normal windows collected. Something is wrong.")

    X_pool = np.stack(all_windows).astype(np.float32)

    out_npz = paths.processed_dir / "train_windows_norm_pool.npz"
    np.savez_compressed(
        out_npz,
        X=X_pool,
        feature_cols=np.array(feature_cols, dtype=object),
        window_size=np.array(window_size),
        stride=np.array(stride),
        source_files=np.array(sources, dtype=object),
    )

    print(f"[INFO] Collected strict-normal windows: {X_pool.shape[0]}")
    print(f"[INFO] Window shape: {X_pool.shape[1:]} (window_size, n_features)")
    print(f"[INFO] Saved pool: {out_npz}")
    print("[SUCCESS] Step 1 complete: built strict-normal window pool (no PCA retrain yet).")


if __name__ == "__main__":
    main()
