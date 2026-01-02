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
    window_size = int(scaler["window_size"])
    stride = int(scaler["stride"])
    mean = np.array(scaler["mean"], dtype=np.float64)
    std = np.array(scaler["std_safe"], dtype=np.float64)

    ts_col = cfg["dataset_prep"]["timestamp_col"]

    thr_w = float(thresholds["threshold_p95"])
    thr_c = float(thresholds["threshold_p99"])

    extracted_repo_dir = download_skab_repo_zip(cfg["skab"]["repo_zip_url"], paths.raw_dir)
    base = (
        extracted_repo_dir
        / cfg["skab"]["extracted_root_dirname"]
        / cfg["skab"]["data_subdir"]
    )

    rel = "anomaly-free/anomaly-free.csv"
    df = load_one_csv(base / rel)

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    values_raw = df[feature_cols].to_numpy(dtype=np.float64)
    values_norm = (values_raw - mean) / std

    scores = score_windows(values_norm, pca, window_size, stride)
    if scores.size == 0:
        raise RuntimeError("No windows produced for anomaly-free file")

    q = np.quantile(scores, [0.0, 0.5, 0.95, 0.99, 1.0])
    frac_w = float((scores > thr_w).mean())
    frac_c = float((scores > thr_c).mean())

    report = {
        "file": rel,
        "n_rows": int(len(df)),
        "window_size": window_size,
        "stride": stride,
        "n_windows": int(len(scores)),
        "threshold_warning_p95": thr_w,
        "threshold_critical_p99": thr_c,
        "score_quantiles": {
            "min": float(q[0]),
            "p50": float(q[1]),
            "p95": float(q[2]),
            "p99": float(q[3]),
            "max": float(q[4]),
        },
        "fraction_windows_above_warning": frac_w,
        "fraction_windows_above_critical": frac_c,
    }

    out_path = Path("reports") / "anomaly_free_ood_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[SUCCESS] Saved: {out_path}")
    print(f"[INFO] anomaly-free score quantiles: {report['score_quantiles']}")
    print(f"[INFO] fraction > warning(p95): {frac_w:.3f}")
    print(f"[INFO] fraction > critical(p99): {frac_c:.3f}")


if __name__ == "__main__":
    main()
