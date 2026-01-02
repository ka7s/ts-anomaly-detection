from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsad.config import load_config, get_paths
from tsad.data.download import download_skab_repo_zip
from tsad.data.load import load_one_csv


def iter_strict_normal_windows_raw(
    values_raw: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int,
) -> list[np.ndarray]:
    windows = []
    n = len(labels)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        if labels[start:end].max() == 0:
            windows.append(values_raw[start:end])
    return windows


def main() -> None:
    cfg = load_config("configs/default.yaml")
    paths = get_paths(cfg)

    manifest_path = paths.processed_dir / "skab_manifest.json"
    scaler_path = paths.processed_dir / "windowing_scaler.json"  # only for feature list + params

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files_with_anomaly = manifest["files_with_anomaly"]

    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    feature_cols = scaler["features"]
    ts_col = scaler["timestamp_col"]
    window_size = int(scaler["window_size"])
    stride = int(scaler["stride"])

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
        df = load_one_csv(base / rel)

        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

        if anomaly_col not in df.columns:
            continue

        # numeric features
        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=feature_cols).reset_index(drop=True)

        labels = pd.to_numeric(df[anomaly_col], errors="coerce").fillna(0).to_numpy(dtype=int)
        values_raw = df[feature_cols].to_numpy(dtype=np.float64)

        windows = iter_strict_normal_windows_raw(values_raw, labels, window_size, stride)
        if windows:
            all_windows.extend(windows)
            sources.extend([rel] * len(windows))

    if not all_windows:
        raise RuntimeError("No strict-normal RAW windows collected.")

    X_raw = np.stack(all_windows).astype(np.float32)

    out_npz = paths.processed_dir / "train_windows_raw_pool.npz"
    np.savez_compressed(
        out_npz,
        X=X_raw,
        feature_cols=np.array(feature_cols, dtype=object),
        window_size=np.array(window_size),
        stride=np.array(stride),
        source_files=np.array(sources, dtype=object),
    )

    print(f"[INFO] Collected strict-normal RAW windows: {X_raw.shape[0]}")
    print(f"[INFO] Window shape: {X_raw.shape[1:]} (window_size, n_features)")
    print(f"[SUCCESS] Saved raw pool: {out_npz}")


if __name__ == "__main__":
    main()
