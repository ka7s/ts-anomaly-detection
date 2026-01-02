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


def score_file_event(
    df: pd.DataFrame,
    ts_col: str,
    feature_cols: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    pca,
    window_size: int,
    stride: int,
    thr: float,
) -> bool:
    # Clean timestamp + sort
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    # Ensure numeric features
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    values_raw = df[feature_cols].to_numpy(dtype=np.float64)
    values_norm = (values_raw - mean) / std

    Xw = make_windows(values_norm, window_size, stride)
    if Xw.shape[0] == 0:
        return False

    n_windows, ws, nf = Xw.shape
    X = Xw.reshape(n_windows, ws * nf).astype(np.float64)

    Z = pca.transform(X)
    X_hat = pca.inverse_transform(Z)
    scores = ((X - X_hat) ** 2).mean(axis=1)

    # Event predicted if ANY window above threshold
    return bool((scores > thr).any())


def main() -> None:
    cfg = load_config("configs/default.yaml")
    paths = get_paths(cfg)

    # Load per-file window eval (for ground-truth event flags)
    eval_csv = Path("reports") / "pca_pool_v2_eval.csv"
    if not eval_csv.exists():
        raise FileNotFoundError(f"Missing eval CSV: {eval_csv}")
    df_eval = pd.read_csv(eval_csv)

    # Ground-truth event per labeled file
    df_eval["has_event_true"] = df_eval["anomalous_windows"] > 0

    # Load model/scaler/thresholds
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
    thr_warning = float(thresholds["threshold_p95"])
    thr_critical = float(thresholds["threshold_p99"])

    # Add anomaly-free file as NO-EVENT ground truth
    extracted_repo_dir = download_skab_repo_zip(cfg["skab"]["repo_zip_url"], paths.raw_dir)
    base = (
        extracted_repo_dir
        / cfg["skab"]["extracted_root_dirname"]
        / cfg["skab"]["data_subdir"]
    )
    anomaly_free_rel = "anomaly-free/anomaly-free.csv"
    df_af = load_one_csv(base / anomaly_free_rel)

    pred_warning_af = score_file_event(
        df_af, ts_col, feature_cols, mean, std, pca, window_size, stride, thr_warning
    )
    pred_critical_af = score_file_event(
        df_af, ts_col, feature_cols, mean, std, pca, window_size, stride, thr_critical
    )

    # Event predictions for labeled files (from window-eval summary)
    # event predicted if TP+FP > 0 => any predicted anomalous window
    df_eval["has_event_pred_warning"] = (df_eval["warning_tp"] + df_eval["warning_fp"]) > 0
    df_eval["has_event_pred_critical"] = (df_eval["critical_tp"] + df_eval["critical_fp"]) > 0

    # Append anomaly-free as one extra row
    extra = pd.DataFrame(
        [{
            "file": anomaly_free_rel,
            "has_event_true": False,
            "has_event_pred_warning": bool(pred_warning_af),
            "has_event_pred_critical": bool(pred_critical_af),
        }]
    )

    df_events = pd.concat(
        [df_eval[["file", "has_event_true", "has_event_pred_warning", "has_event_pred_critical"]], extra],
        ignore_index=True,
    )

    def event_cm(true: pd.Series, pred: pd.Series) -> dict[str, int]:
        tp = int((true & pred).sum())
        fp = int((~true & pred).sum())
        fn = int((true & ~pred).sum())
        tn = int((~true & ~pred).sum())
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def prf(cm: dict[str, int]) -> dict[str, float]:
        tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}

    cm_w = event_cm(df_events["has_event_true"], df_events["has_event_pred_warning"])
    cm_c = event_cm(df_events["has_event_true"], df_events["has_event_pred_critical"])

    out = {
        "n_files_including_anomaly_free": int(len(df_events)),
        "threshold_warning_p95": thr_warning,
        "threshold_critical_p99": thr_critical,
        "anomaly_free_pred_warning": bool(pred_warning_af),
        "anomaly_free_pred_critical": bool(pred_critical_af),
        "event_warning": {**cm_w, **prf(cm_w)},
        "event_critical": {**cm_c, **prf(cm_c)},
    }

    out_path = Path("reports") / "pca_pool_v2_event_with_anomaly_free.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[SUCCESS] Saved: {out_path}")
    print(f"[INFO] anomaly-free predicted event? warning={pred_warning_af}, critical={pred_critical_af}")
    print(f"[INFO] Event-level WARNING: {out['event_warning']}")
    print(f"[INFO] Event-level CRITICAL: {out['event_critical']}")


if __name__ == "__main__":
    main()
