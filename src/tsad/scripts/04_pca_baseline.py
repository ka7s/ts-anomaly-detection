from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.decomposition import PCA

from tsad.config import load_config, get_paths


def reconstruction_mse(X: np.ndarray, X_hat: np.ndarray) -> np.ndarray:
    """Return per-sample MSE (row-wise)."""
    err = (X - X_hat) ** 2
    return err.mean(axis=1)


def main() -> None:
    cfg = load_config("configs/default.yaml")
    paths = get_paths(cfg)

    # Inputs
    npz_path = paths.processed_dir / "train_windows_norm.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing windows file: {npz_path}")

    # Outputs
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    out_model = models_dir / "pca.joblib"
    out_scores = paths.processed_dir / "pca_train_scores.csv"
    out_thresholds = paths.processed_dir / "pca_thresholds.json"

    # Load windows
    data = np.load(npz_path, allow_pickle=True)
    Xw = data["X"]  # (n_windows, window_size, n_features)

    if np.isnan(Xw).any() or np.isinf(Xw).any():
        raise ValueError("NaN/Inf detected in X windows")

    n_windows, window_size, n_features = Xw.shape
    X = Xw.reshape(n_windows, window_size * n_features)

    print(f"[INFO] Loaded: {npz_path}")
    print(f"[INFO] Flattened X shape: {X.shape} (n_windows, window_size*n_features)")

    # -----------------------------
    # Train PCA baseline
    # -----------------------------
    # Keep 95% variance — PCA decides number of components.
    pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
    Z = pca.fit_transform(X)
    X_hat = pca.inverse_transform(Z)

    scores = reconstruction_mse(X, X_hat)

    print(f"[INFO] PCA chose n_components={pca.n_components_} to keep ~95% variance")
    print(f"[INFO] Train reconstruction MSE: mean={scores.mean():.6f}, std={scores.std():.6f}")
    print(f"[INFO] Score quantiles: p95={np.quantile(scores, 0.95):.6f}, p99={np.quantile(scores, 0.99):.6f}")

    # -----------------------------
    # Save artifacts
    # -----------------------------
    dump(pca, out_model)

    df_scores = pd.DataFrame(
        {
            "window_index": np.arange(n_windows, dtype=int),
            "recon_mse": scores.astype(float),
        }
    )
    df_scores.to_csv(out_scores, index=False)

    thresholds = {
        "method": "PCA reconstruction MSE",
        "train_windows": int(n_windows),
        "window_size": int(window_size),
        "n_features": int(n_features),
        "flatten_dim": int(window_size * n_features),
        "n_components": int(pca.n_components_),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "threshold_mean_plus_3std": float(scores.mean() + 3.0 * scores.std()),
        "threshold_p95": float(np.quantile(scores, 0.95)),
        "threshold_p99": float(np.quantile(scores, 0.99)),
    }

    out_thresholds.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    print(f"[SUCCESS] Saved model: {out_model}")
    print(f"[SUCCESS] Saved train scores: {out_scores}")
    print(f"[SUCCESS] Saved thresholds: {out_thresholds}")


if __name__ == "__main__":
    main()
