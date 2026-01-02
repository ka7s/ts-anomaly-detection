from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tsad.config import load_config, get_paths


def main() -> None:
    cfg = load_config("configs/default.yaml")
    paths = get_paths(cfg)

    pool_path = paths.processed_dir / "train_windows_norm_pool.npz"
    if not pool_path.exists():
        raise FileNotFoundError(f"Missing pool file: {pool_path}")

    data = np.load(pool_path, allow_pickle=True)
    X = data["X"].astype(np.float64)  # (n_windows, window_size, n_features)
    feature_cols = data["feature_cols"].tolist()
    window_size = int(data["window_size"])
    stride = int(data["stride"])

    n_windows, ws, n_features = X.shape
    flat = X.reshape(-1, n_features)

    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std_safe = np.where(std < 1e-12, 1.0, std)

    out_path = paths.processed_dir / "pool_scaler.json"

    payload = {
        "method": "z-score scaler fitted on strict-normal pool windows",
        "features": feature_cols,
        "window_size": window_size,
        "stride": stride,
        "n_windows": int(n_windows),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "std_safe": std_safe.tolist(),
        "dtype": "float64",
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("[INFO] Pool-scaler stats:")
    for name, m, s in zip(feature_cols, mean, std):
        print(f"  {name:>22s}  mean={m:.6f}  std={s:.6f}")

    print(f"[SUCCESS] Saved pool scaler: {out_path}")


if __name__ == "__main__":
    main()
