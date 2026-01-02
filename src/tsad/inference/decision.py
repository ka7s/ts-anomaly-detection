# src/tsad/inference/decision.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import numpy as np
import pandas as pd
from joblib import load

from tsad.config import load_config, get_paths
from tsad.data.download import download_skab_repo_zip
from tsad.data.load import load_one_csv


def _make_windows(values: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    windows = []
    for start in range(0, len(values) - window_size + 1, stride):
        windows.append(values[start:start + window_size])
    return np.stack(windows) if windows else np.empty((0, window_size, values.shape[1]), dtype=values.dtype)


def _consecutive_k(flags: np.ndarray, k: int) -> bool:
    count = 0
    for f in flags:
        count = count + 1 if f else 0
        if count >= k:
            return True
    return False


def _score_file_windows(
    df: pd.DataFrame,
    ts_col: str,
    feature_cols: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    pca,
    window_size: int,
    stride: int,
) -> np.ndarray:
    df = df.copy()

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    X_raw = df[feature_cols].to_numpy(dtype=np.float64)
    X_norm = (X_raw - mean) / std

    Xw = _make_windows(X_norm, window_size, stride)
    if Xw.shape[0] == 0:
        return np.array([], dtype=np.float64)

    n, ws, nf = Xw.shape
    X = Xw.reshape(n, ws * nf).astype(np.float64)

    Z = pca.transform(X)
    X_hat = pca.inverse_transform(Z)
    scores = ((X - X_hat) ** 2).mean(axis=1)
    return scores


def _train_p99_from_scores_csv(train_scores_csv: Path) -> float:
    df = pd.read_csv(train_scores_csv)
    if "recon_mse" not in df.columns:
        raise ValueError("Expected column 'recon_mse' in train score CSV")
    return float(np.quantile(df["recon_mse"].to_numpy(dtype=float), 0.99))


@dataclass(frozen=True)
class Artifacts:
    pca_model_path: Path
    thresholds_path: Path
    scaler_path: Path
    train_scores_path: Path


def default_artifacts(paths) -> Artifacts:
    return Artifacts(
        pca_model_path=Path("models") / "pca_pool_v2.joblib",
        thresholds_path=paths.processed_dir / "pca_pool_v2_thresholds.json",
        scaler_path=paths.processed_dir / "raw_pool_scaler.json",
        train_scores_path=paths.processed_dir / "pca_pool_v2_train_scores.csv",
    )


def decide_skab_file(
    rel_path: str,
    warning_k: int = 3,
    critical_k: int = 5,
    ood_margin: float = 0.0,
    config_path: str = "configs/default.yaml",
) -> dict[str, Any]:
    """
    Decide OOD + warning/critical for one SKAB file (relative path under SKAB data/).

    Returns a JSON-serializable dict with:
      - ood (bool)
      - warning (bool)
      - critical (bool)
      - score quantiles
      - thresholds + windowing config
    """
    if warning_k <= 0 or critical_k <= 0:
        raise ValueError("warning_k and critical_k must be positive integers")
    if critical_k < warning_k:
        raise ValueError("critical_k should be >= warning_k")

    cfg = load_config(config_path)
    paths = get_paths(cfg)

    artifacts = default_artifacts(paths)

    # Load artifacts
    scaler = json.loads(artifacts.scaler_path.read_text(encoding="utf-8"))
    thresholds = json.loads(artifacts.thresholds_path.read_text(encoding="utf-8"))
    pca = load(artifacts.pca_model_path)

    feature_cols = scaler["features"]
    mean = np.array(scaler["mean"], dtype=np.float64)
    std = np.array(scaler["std_safe"], dtype=np.float64)
    window_size = int(scaler["window_size"])
    stride = int(scaler["stride"])

    ts_col = cfg["dataset_prep"]["timestamp_col"]

    thr_warning = float(thresholds["threshold_p95"])
    thr_critical = float(thresholds["threshold_p99"])
    train_p99 = _train_p99_from_scores_csv(artifacts.train_scores_path)

    # Locate data file inside SKAB
    
    env_data_root = os.getenv("TSAD_DATA_ROOT")

    if env_data_root:
        base = Path(env_data_root)
    else:
        extracted_repo_dir = download_skab_repo_zip(cfg["skab"]["repo_zip_url"], paths.raw_dir)
        base = extracted_repo_dir / cfg["skab"]["extracted_root_dirname"] / cfg["skab"]["data_subdir"]

    df = load_one_csv(base / rel_path)

    

    scores = _score_file_windows(
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
        raise RuntimeError(f"No windows produced for: {rel_path}")

    file_min = float(scores.min())
    file_p50 = float(np.quantile(scores, 0.50))
    file_p95 = float(np.quantile(scores, 0.95))
    file_p99 = float(np.quantile(scores, 0.99))
    file_max = float(scores.max())

    # OOD gate uses median vs train_p99 (+ margin)
    ood = file_p50 > (train_p99 + float(ood_margin))

    flags_warning = scores > thr_warning
    flags_critical = scores > thr_critical

    warning = (not ood) and _consecutive_k(flags_warning, warning_k)
    critical = (not ood) and _consecutive_k(flags_critical, critical_k)

    return {
        "file": rel_path,
        "windowing": {"window_size": window_size, "stride": stride, "n_windows": int(len(scores))},
        "thresholds": {
            "warning_p95": thr_warning,
            "critical_p99": thr_critical,
            "ood_train_p99": train_p99,
            "ood_margin": float(ood_margin),
            "warning_k": int(warning_k),
            "critical_k": int(critical_k),
        },
        "scores": {"min": file_min, "p50": file_p50, "p95": file_p95, "p99": file_p99, "max": file_max},
        "ood": bool(ood),
        "warning": bool(warning),
        "critical": bool(critical),
        "model": {
            "pca_model_path": str(artifacts.pca_model_path),
            "scaler_path": str(artifacts.scaler_path),
            "thresholds_path": str(artifacts.thresholds_path),
            "train_scores_path": str(artifacts.train_scores_path),
        },
    }
