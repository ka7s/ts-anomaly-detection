# src/tsad/scripts/09_eval_pca_pool_v2_all_files.py
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
    # window anomalous if ANY point in window is 1
    y = []
    n = len(labels)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        y.append(int(labels[start:end].max() > 0))
    return np.array(y, dtype=int)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def main() -> None:
    cfg = load_config("configs/default.yaml")
    paths = get_paths(cfg)

    # --- Artifacts (pool v2) ---
    scaler_path = paths.processed_dir / "raw_pool_scaler.json"
    thresholds_path = paths.processed_dir / "pca_pool_v2_thresholds.json"
    model_path = Path("models") / "pca_pool_v2.joblib"

    manifest_path = paths.processed_dir / "skab_manifest.json"

    for p in [scaler_path, thresholds_path, model_path, manifest_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pca = load(model_path)

    feature_cols = scaler["features"]
    window_size = int(scaler["window_size"])
    stride = int(scaler["stride"])
    mean = np.array(scaler["mean"], dtype=np.float64)
    std = np.array(scaler["std_safe"], dtype=np.float64)

    ts_col = cfg["dataset_prep"]["timestamp_col"]
    anomaly_col = cfg["dataset_prep"]["anomaly_col"]

    thr_warning = float(thresholds["threshold_p95"])
    thr_critical = float(thresholds["threshold_p99"])

    files_with_anomaly = manifest["files_with_anomaly"]

    # Ensure SKAB is present
    extracted_repo_dir = download_skab_repo_zip(cfg["skab"]["repo_zip_url"], paths.raw_dir)
    base = (
        extracted_repo_dir
        / cfg["skab"]["extracted_root_dirname"]
        / cfg["skab"]["data_subdir"]
    )

    rows = []

    # Overall accumulators
    overall = {
        "warning": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "critical": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    }

    print(f"[INFO] Evaluating {len(files_with_anomaly)} labeled files")
    print(f"[INFO] window_size={window_size}, stride={stride}")
    print(f"[INFO] thresholds: warning(p95)={thr_warning:.6f}, critical(p99)={thr_critical:.6f}")

    for rel in files_with_anomaly:
        df = load_one_csv(base / rel)

        # timestamp clean + sort
        if ts_col not in df.columns:
            print(f"[WARN] Skipping {rel}: missing timestamp col '{ts_col}'")
            continue

        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

        # anomaly col
        if anomaly_col not in df.columns:
            print(f"[WARN] Skipping {rel}: missing anomaly col '{anomaly_col}'")
            continue

        # numeric features
        missing_features = [c for c in feature_cols if c not in df.columns]
        if missing_features:
            print(f"[WARN] Skipping {rel}: missing features {missing_features}")
            continue

        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=feature_cols).reset_index(drop=True)

        labels = pd.to_numeric(df[anomaly_col], errors="coerce").fillna(0).to_numpy(dtype=int)
        values_raw = df[feature_cols].to_numpy(dtype=np.float64)
        values_norm = (values_raw - mean) / std

        Xw = make_windows(values_norm, window_size, stride)
        if Xw.shape[0] == 0:
            print(f"[WARN] Skipping {rel}: no windows could be formed")
            continue

        y_win = make_window_labels(labels, window_size, stride)

        n_windows, ws, nf = Xw.shape
        X = Xw.reshape(n_windows, ws * nf).astype(np.float64)

        Z = pca.transform(X)
        X_hat = pca.inverse_transform(Z)
        scores = ((X - X_hat) ** 2).mean(axis=1)

        # Predictions
        pred_warning = (scores > thr_warning).astype(int)
        pred_critical = (scores > thr_critical).astype(int)

        cm_w = confusion(y_win, pred_warning)
        cm_c = confusion(y_win, pred_critical)

        # Aggregate overall
        for k in overall["warning"]:
            overall["warning"][k] += cm_w[k]
            overall["critical"][k] += cm_c[k]

        # Per-file metrics
        met_w = prf(cm_w["tp"], cm_w["fp"], cm_w["fn"])
        met_c = prf(cm_c["tp"], cm_c["fp"], cm_c["fn"])

        rows.append(
            {
                "file": rel,
                "rows": int(len(df)),
                "windows": int(n_windows),
                "anomaly_rate_points": float(labels.mean()),
                "anomalous_windows": int(y_win.sum()),
                # warning
                "warning_tp": cm_w["tp"],
                "warning_fp": cm_w["fp"],
                "warning_fn": cm_w["fn"],
                "warning_tn": cm_w["tn"],
                "warning_precision": met_w["precision"],
                "warning_recall": met_w["recall"],
                "warning_f1": met_w["f1"],
                # critical
                "critical_tp": cm_c["tp"],
                "critical_fp": cm_c["fp"],
                "critical_fn": cm_c["fn"],
                "critical_tn": cm_c["tn"],
                "critical_precision": met_c["precision"],
                "critical_recall": met_c["recall"],
                "critical_f1": met_c["f1"],
            }
        )

    if not rows:
        raise RuntimeError("No files evaluated. Check dataset paths and columns.")

    df_eval = pd.DataFrame(rows)

    # Save outputs
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    out_csv = reports_dir / "pca_pool_v2_eval.csv"
    out_json = reports_dir / "pca_pool_v2_summary.json"

    df_eval.to_csv(out_csv, index=False)

    summary = {
        "n_files_evaluated": int(len(df_eval)),
        "window_size": window_size,
        "stride": stride,
        "threshold_warning_p95": thr_warning,
        "threshold_critical_p99": thr_critical,
        "overall_warning": {
            **overall["warning"],
            **prf(overall["warning"]["tp"], overall["warning"]["fp"], overall["warning"]["fn"]),
        },
        "overall_critical": {
            **overall["critical"],
            **prf(overall["critical"]["tp"], overall["critical"]["fp"], overall["critical"]["fn"]),
        },
    }

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Print compact summary
    ow = summary["overall_warning"]
    oc = summary["overall_critical"]
    print(f"[SUCCESS] Saved per-file eval: {out_csv}")
    print(f"[SUCCESS] Saved summary: {out_json}")
    print("[INFO] Overall (warning/p95): "
          f"P={ow['precision']:.3f} R={ow['recall']:.3f} F1={ow['f1']:.3f} "
          f"(TP={ow['tp']}, FP={ow['fp']}, FN={ow['fn']}, TN={ow['tn']})")
    print("[INFO] Overall (critical/p99): "
          f"P={oc['precision']:.3f} R={oc['recall']:.3f} F1={oc['f1']:.3f} "
          f"(TP={oc['tp']}, FP={oc['fp']}, FN={oc['fn']}, TN={oc['tn']})")


if __name__ == "__main__":
    main()
