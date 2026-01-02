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
    n = len(values)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        windows.append(values[start:end])
    if not windows:
        return np.empty((0, window_size, values.shape[1]), dtype=values.dtype)
    return np.stack(windows)


def make_window_labels(labels: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    y = []
    n = len(labels)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        y.append(int(labels[start:end].max() > 0))
    return np.array(y, dtype=int)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return tp, fp, fn, tn


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main() -> None:
    cfg = load_config("configs/default.yaml")
    paths = get_paths(cfg)

    scaler_path = paths.processed_dir / "raw_pool_scaler.json"
    thresholds_path = paths.processed_dir / "pca_pool_v2_thresholds.json"
    model_path = Path("models") / "pca_pool_v2.joblib"

    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    pca = load(model_path)

    feature_cols = scaler["features"]
    ts_col = cfg["dataset_prep"]["timestamp_col"]
    anomaly_col = cfg["dataset_prep"]["anomaly_col"]

    window_size = int(scaler["window_size"])
    stride = int(scaler["stride"])
    mean = np.array(scaler["mean"], dtype=np.float64)
    std = np.array(scaler["std_safe"], dtype=np.float64)

    rel = "valve1/1.csv"
    print(f"[INFO] Scoring labeled file: {rel}")
    print(f"[INFO] Using model: {model_path}")

    extracted_repo_dir = download_skab_repo_zip(cfg["skab"]["repo_zip_url"], paths.raw_dir)
    base = (
        extracted_repo_dir
        / cfg["skab"]["extracted_root_dirname"]
        / cfg["skab"]["data_subdir"]
    )
    df = load_one_csv(base / rel)

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Missing feature '{c}' in {rel}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    if anomaly_col not in df.columns:
        raise ValueError(f"No anomaly column '{anomaly_col}' in {rel}")

    labels = pd.to_numeric(df[anomaly_col], errors="coerce").fillna(0).to_numpy(dtype=int)

    values_raw = df[feature_cols].to_numpy(dtype=np.float64)
    values_norm = (values_raw - mean) / std

    Xw = make_windows(values_norm, window_size=window_size, stride=stride)
    y_win = make_window_labels(labels, window_size=window_size, stride=stride)

    n_windows, ws, nf = Xw.shape
    X = Xw.reshape(n_windows, ws * nf).astype(np.float64)

    Z = pca.transform(X)
    X_hat = pca.inverse_transform(Z)
    scores = ((X - X_hat) ** 2).mean(axis=1)

    q = np.quantile(scores, [0.0, 0.5, 0.95, 0.99, 1.0])
    print(f"[INFO] Windows: {n_windows}  (window_size={window_size}, stride={stride})")
    print(f"[INFO] Score quantiles: min={q[0]:.6f}, p50={q[1]:.6f}, p95={q[2]:.6f}, p99={q[3]:.6f}, max={q[4]:.6f}")
    print(f"[INFO] Window-level anomalous windows (ANY=1): {y_win.sum()} / {len(y_win)}")

    # Threshold sweep (single-file tuning)
    candidates = {
        "p95_train": float(thresholds["threshold_p95"]),
        "p99_train": float(thresholds["threshold_p99"]),
        "mean+3std_train": float(thresholds["threshold_mean_plus_3std"]),
        # extra useful ones between 95 and 99
        "p97_file": float(np.quantile(scores, 0.97)),
        "p98_file": float(np.quantile(scores, 0.98)),
    }

    print("[INFO] Threshold sweep:")
    for name, thr in candidates.items():
        y_pred = (scores > thr).astype(int)
        tp, fp, fn, tn = confusion(y_win, y_pred)
        precision, recall, f1 = prf(tp, fp, fn)
        print(
            f"  {name:>14s} thr={thr:.6f} | TP={tp:3d} FP={fp:3d} FN={fn:3d} TN={tn:3d} | "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
        )

    print("[SUCCESS] Step complete: threshold sweep on one file.")


if __name__ == "__main__":
    main()
