# src/tsad/scripts/14_decide_file.py
from __future__ import annotations

import argparse
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
    return np.stack(windows) if windows else np.empty((0, window_size, values.shape[1]), dtype=values.dtype)


def consecutive_k(flags: np.ndarray, k: int) -> bool:
    count = 0
    for f in flags:
        count = count + 1 if f else 0
        if count >= k:
            return True
    return False


def score_file_windows(
    df: pd.DataFrame,
    ts_col: str,
    feature_cols: list[str],
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
    if Xw.shape[0] == 0:
        return np.array([], dtype=np.float64)

    n, ws, nf = Xw.shape
    X = Xw.reshape(n, ws * nf).astype(np.float64)

    Z = pca.transform(X)
    X_hat = pca.inverse_transform(Z)
    scores = ((X - X_hat) ** 2).mean(axis=1)
    return scores


def load_train_p99(paths) -> float:
    train_scores_csv = paths.processed_dir / "pca_pool_v2_train_scores.csv"
    if not train_scores_csv.exists():
        raise FileNotFoundError(f"Missing: {train_scores_csv}")
    df = pd.read_csv(train_scores_csv)
    if "recon_mse" not in df.columns:
        raise ValueError("Expected column 'recon_mse' in pca_pool_v2_train_scores.csv")
    return float(np.quantile(df["recon_mse"].to_numpy(dtype=float), 0.99))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decide OOD + warning/critical alerts for one SKAB file (split persistence)."
    )
    parser.add_argument("--rel", type=str, default="valve1/1.csv", help="Relative path inside SKAB data/ folder")
    parser.add_argument("--warning-k", type=int, default=3, help="Persistence for warning threshold (consecutive)")
    parser.add_argument("--critical-k", type=int, default=5, help="Persistence for critical threshold (consecutive)")
    parser.add_argument("--margin", type=float, default=0.0, help="OOD margin added to train p99 gate")
    args = parser.parse_args()

    if args.warning_k <= 0 or args.critical_k <= 0:
        raise ValueError("warning-k and critical-k must be positive integers")
    if args.critical_k < args.warning_k:
        # Not strictly required, but usually the right policy.
        raise ValueError("critical-k should be >= warning-k (critical should be stricter)")

    cfg = load_config("configs/default.yaml")
    paths = get_paths(cfg)

    scaler = json.loads((paths.processed_dir / "raw_pool_scaler.json").read_text(encoding="utf-8"))
    thresholds = json.loads((paths.processed_dir / "pca_pool_v2_thresholds.json").read_text(encoding="utf-8"))
    pca = load(Path("models") / "pca_pool_v2.joblib")

    feature_cols = scaler["features"]
    mean = np.array(scaler["mean"], dtype=np.float64)
    std = np.array(scaler["std_safe"], dtype=np.float64)
    window_size = int(scaler["window_size"])
    stride = int(scaler["stride"])

    ts_col = cfg["dataset_prep"]["timestamp_col"]

    thr_warning = float(thresholds["threshold_p95"])
    thr_critical = float(thresholds["threshold_p99"])

    train_p99 = load_train_p99(paths)

    extracted_repo_dir = download_skab_repo_zip(cfg["skab"]["repo_zip_url"], paths.raw_dir)
    base = extracted_repo_dir / cfg["skab"]["extracted_root_dirname"] / cfg["skab"]["data_subdir"]

    rel = args.rel
    df = load_one_csv(base / rel)

    scores = score_file_windows(
        df=df,
        ts_col=ts_col,
        feature_cols=feature_cols,
        mean=mean,
        std=std,
        pca=pca,
        window_size=window_size,
        stride=stride,
    )
    if scores.size == 0:
        raise RuntimeError(f"No windows produced for: {rel}")

    file_p50 = float(np.quantile(scores, 0.50))
    file_p95 = float(np.quantile(scores, 0.95))
    file_p99 = float(np.quantile(scores, 0.99))
    file_min = float(scores.min())
    file_max = float(scores.max())

    # --- OOD gate ---
    ood_gate = file_p50 > (train_p99 + args.margin)

    # --- alerts (only meaningful if IN_DOMAIN) ---
    flags_warning = scores > thr_warning
    flags_critical = scores > thr_critical

    warning = (not ood_gate) and consecutive_k(flags_warning, args.warning_k)
    critical = (not ood_gate) and consecutive_k(flags_critical, args.critical_k)

    decision = {
        "file": rel,
        "windowing": {"window_size": window_size, "stride": stride, "n_windows": int(len(scores))},
        "thresholds": {
            "warning_p95": thr_warning,
            "critical_p99": thr_critical,
            "ood_train_p99": train_p99,
            "ood_margin": float(args.margin),
            "warning_k": int(args.warning_k),
            "critical_k": int(args.critical_k),
        },
        "scores": {
            "min": file_min,
            "p50": file_p50,
            "p95": file_p95,
            "p99": file_p99,
            "max": file_max,
        },
        "ood": bool(ood_gate),
        "warning": bool(warning),
        "critical": bool(critical),
    }

    out_path = Path("reports") / f"decision_{Path(rel).as_posix().replace('/', '__')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    print(f"[SUCCESS] Saved decision: {out_path}")
    print(f"[INFO] OOD={decision['ood']}  WARNING={decision['warning']}  CRITICAL={decision['critical']}")
    print(
        f"[INFO] score p50={file_p50:.6f} vs train_p99={train_p99:.6f} (margin={args.margin:.6f}) | "
        f"warning_k={args.warning_k} critical_k={args.critical_k}"
    )


if __name__ == "__main__":
    main()
