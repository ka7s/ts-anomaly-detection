from __future__ import annotations

import json
import numpy as np

from tsad.config import load_config, get_paths


def main() -> None:
    cfg = load_config("configs/default.yaml")
    paths = get_paths(cfg)

    raw_pool_path = paths.processed_dir / "train_windows_raw_pool.npz"
    scaler_path = paths.processed_dir / "raw_pool_scaler.json"

    if not raw_pool_path.exists():
        raise FileNotFoundError(f"Missing raw pool: {raw_pool_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing raw pool scaler: {scaler_path}")

    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    mean = np.array(scaler["mean"], dtype=np.float64)
    std = np.array(scaler["std_safe"], dtype=np.float64)
    feature_cols = scaler["features"]
    window_size = int(scaler["window_size"])
    stride = int(scaler["stride"])

    data = np.load(raw_pool_path, allow_pickle=True)
    X_raw = data["X"].astype(np.float64)
    sources = data["source_files"]

    if np.isnan(X_raw).any() or np.isinf(X_raw).any():
        raise ValueError("NaN/Inf detected in RAW pool")

    # Normalize with broadcasting over last dimension
    X_norm = ((X_raw - mean) / std).astype(np.float32)

    if np.isnan(X_norm).any() or np.isinf(X_norm).any():
        raise ValueError("NaN/Inf detected after normalization")

    # Sanity check: mean/std over entire normalized pool
    flat = X_norm.reshape(-1, X_norm.shape[-1]).astype(np.float64)
    mean_norm = flat.mean(axis=0)
    std_norm = flat.std(axis=0)

    print("[INFO] Normalization check on pool (should be ~0 mean, ~1 std):")
    for name, m, s in zip(feature_cols, mean_norm, std_norm):
        print(f"  {name:>22s}  mean={m:.6f}  std={s:.6f}")

    out_path = paths.processed_dir / "train_windows_norm_pool_v2.npz"
    np.savez_compressed(
        out_path,
        X=X_norm,
        feature_cols=np.array(feature_cols, dtype=object),
        window_size=np.array(window_size),
        stride=np.array(stride),
        source_files=sources,
    )

    print(f"[SUCCESS] Saved normalized pool v2: {out_path}")
    print(f"[INFO] Shape: {X_norm.shape}, dtype={X_norm.dtype}")


if __name__ == "__main__":
    main()
