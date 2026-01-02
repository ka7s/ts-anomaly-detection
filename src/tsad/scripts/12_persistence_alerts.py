from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from tsad.config import load_config, get_paths
from tsad.data.download import download_skab_repo_zip
from tsad.data.load import load_one_csv


def make_windows(values: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    windows = []
    for start in range(0, len(values) - window_size + 1, stride):
        windows.append(values[start:start + window_size])
    return np.stack(windows) if windows else np.empty((0, window_size, values.shape[1]))


def consecutive_k(flags: np.ndarray, k: int) -> bool:
    """Return True if there are k consecutive True values."""
    count = 0
    for f in flags:
        count = count + 1 if f else 0
        if count >= k:
            return True
    return False


def score_file(
    df: pd.DataFrame,
    feature_cols: list[str],
    ts_col: str,
    mean: np.ndarray,
    std: np.ndarray,
    pca,
    window_size: int,
    stride: int,
) -> np.ndarray:
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    X_raw = df[feature_cols].to_numpy(dtype=np.float64)
    X_norm = (X_raw - mean) / std

    Xw = make_windows(X_norm, window_size, stride)
    n, ws, nf = Xw.shape
    X = Xw.reshape(n, ws * nf)

    Z = pca.transform(X)
    X_hat = pca.inverse_transform(Z)
    scores = ((X - X_hat) ** 2).mean(axis=1)
    return scores


def main() -> None:
    cfg = load_config("configs/default.yaml")
    paths = get_paths(cfg)

    scaler = json.loads((paths.processed_dir / "raw_pool_scaler.json").read_text())
    thresholds = json.loads((paths.processed_dir / "pca_pool_v2_thresholds.json").read_text())
    pca = load(Path("models") / "pca_pool_v2.joblib")

    feature_cols = scaler["features"]
    mean = np.array(scaler["mean"])
    std = np.array(scaler["std_safe"])
    window_size = int(scaler["window_size"])
    stride = int(scaler["stride"])
    ts_col = cfg["dataset_prep"]["timestamp_col"]

    thr = float(thresholds["threshold_p95"])

    extracted_repo_dir = download_skab_repo_zip(cfg["skab"]["repo_zip_url"], paths.raw_dir)
    base = extracted_repo_dir / cfg["skab"]["extracted_root_dirname"] / cfg["skab"]["data_subdir"]

    test_files = {
        "labeled": "valve1/1.csv",
        "anomaly_free": "anomaly-free/anomaly-free.csv",
    }

    print(f"[INFO] Threshold (warning / p95): {thr:.6f}")
    print(f"[INFO] window_size={window_size}, stride={stride}\n")

    for name, rel in test_files.items():
        df = load_one_csv(base / rel)
        scores = score_file(
            df,
            feature_cols,
            ts_col,
            mean,
            std,
            pca,
            window_size,
            stride,
        )

        flags = scores > thr

        print(f"=== {name.upper()} | {rel} ===")
        print(f"Windows: {len(scores)}")
        print(f"Score quantiles: "
              f"min={scores.min():.4f}, "
              f"p50={np.quantile(scores,0.5):.4f}, "
              f"p95={np.quantile(scores,0.95):.4f}, "
              f"max={scores.max():.4f}")

        for k in [1, 3, 5]:
            fired = consecutive_k(flags, k)
            print(f"Alert fired with consecutive_k={k}: {fired}")

        print()


if __name__ == "__main__":
    main()
