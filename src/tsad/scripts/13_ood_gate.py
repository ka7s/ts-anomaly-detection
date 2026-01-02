# src/tsad/scripts/13_ood_gate.py
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


def score_windows(values_norm: np.ndarray, pca, window_size: int, stride: int) -> np.ndarray:
    Xw = make_windows(values_norm, window_size, stride)
    if Xw.shape[0] == 0:
        return np.array([], dtype=np.float64)
    n, ws, nf = Xw.shape
    X = Xw.reshape(n, ws * nf).astype(np.float64)
    Z = pca.transform(X)
    X_hat = pca.inverse_transform(Z)
    scores = ((X - X_hat) ** 2).mean(axis=1)
    return scores


def load_train_score_stats(train_scores_csv: Path) -> dict[str, float]:
    df = pd.read_csv(train_scores_csv)
    if "recon_mse" not in df.columns:
        raise ValueError("train score csv must contain column 'recon_mse'")
    s = df["recon_mse"].to_numpy(dtype=float)

    return {
        "train_min": float(np.min(s)),
        "train_p50": float(np.quantile(s, 0.50)),
        "train_p95": float(np.quantile(s, 0.95)),
        "train_p99": float(np.quantile(s, 0.99)),
        "train_max": float(np.max(s)),
        "train_mean": float(np.mean(s)),
        "train_std": float(np.std(s)),
    }


def gate_decision(file_median: float, train_p99: float, margin: float = 0.0) -> str:
    # Simple rule: if the file's median score is above train p99 (+ margin), treat as OOD.
    return "OOD" if file_median > (train_p99 + margin) else "IN_DOMAIN"


def main() -> None:
    cfg = load_config("configs/default.yaml")
    paths = get_paths(cfg)

    scaler = json.loads((paths.processed_dir / "raw_pool_scaler.json").read_text())
    pca = load(Path("models") / "pca_pool_v2.joblib")

    # IMPORTANT: use the PCA pool v2 train score distribution
    train_scores_csv = paths.processed_dir / "pca_pool_v2_train_scores.csv"
    if not train_scores_csv.exists():
        raise FileNotFoundError(f"Missing: {train_scores_csv}")

    stats = load_train_score_stats(train_scores_csv)

    feature_cols = scaler["features"]
    mean = np.array(scaler["mean"], dtype=np.float64)
    std = np.array(scaler["std_safe"], dtype=np.float64)
    window_size = int(scaler["window_size"])
    stride = int(scaler["stride"])
    ts_col = cfg["dataset_prep"]["timestamp_col"]

    # Optional safety margin (helps avoid borderline OOD calls)
    margin = 0.0

    extracted_repo_dir = download_skab_repo_zip(cfg["skab"]["repo_zip_url"], paths.raw_dir)
    base = extracted_repo_dir / cfg["skab"]["extracted_root_dirname"] / cfg["skab"]["data_subdir"]

    files = [
        "valve1/1.csv",
        "anomaly-free/anomaly-free.csv",
    ]

    print("[INFO] Train score stats (PCA pool v2):")
    print(
        f"  p50={stats['train_p50']:.6f}  p95={stats['train_p95']:.6f}  "
        f"p99={stats['train_p99']:.6f}  max={stats['train_max']:.6f}"
    )
    print(f"[INFO] Gate rule: OOD if file_median > train_p99 + margin (margin={margin:.6f})\n")

    for rel in files:
        df = load_one_csv(base / rel)

        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

        for c in feature_cols:
            if c not in df.columns:
                raise ValueError(f"Missing feature '{c}' in {rel}")
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=feature_cols).reset_index(drop=True)

        X_raw = df[feature_cols].to_numpy(dtype=np.float64)
        X_norm = (X_raw - mean) / std

        scores = score_windows(X_norm, pca, window_size, stride)
        if scores.size == 0:
            raise RuntimeError(f"No windows produced for: {rel}")

        file_median = float(np.quantile(scores, 0.50))
        file_p95 = float(np.quantile(scores, 0.95))
        file_p99 = float(np.quantile(scores, 0.99))
        file_min = float(np.min(scores))
        file_max = float(np.max(scores))

        decision = gate_decision(file_median, stats["train_p99"], margin=margin)

        print(f"=== {rel} ===")
        print(f"[INFO] score quantiles: min={file_min:.6f} p50={file_median:.6f} p95={file_p95:.6f} p99={file_p99:.6f} max={file_max:.6f}")
        print(f"[INFO] OOD gate decision: {decision}\n")


if __name__ == "__main__":
    main()
