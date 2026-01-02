from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsad.config import load_config, get_paths


def main() -> None:
    cfg = load_config("configs/default.yaml")
    paths = get_paths(cfg)

    parquet_path = paths.processed_dir / "train_normal.parquet"
    meta_path = paths.processed_dir / "train_normal_meta.json"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing parquet file: {parquet_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ts_col = meta["timestamp_col"]
    feature_cols = meta["features"]

    df = pd.read_parquet(parquet_path)
    df = df.sort_values(ts_col).reset_index(drop=True)

    print(f"[INFO] Loaded normal training data: {parquet_path}")
    print(f"[INFO] Rows: {len(df)}, Features: {len(feature_cols)}")

    # -----------------------------
    # Windowing parameters
    # -----------------------------
    window_size = 120
    stride = 12
    print(f"[INFO] Windowing with window_size={window_size}, stride={stride}")

    # -----------------------------
    # Generate windows
    # -----------------------------
    values = df[feature_cols].to_numpy()
    n_samples, n_features = values.shape

    windows = []
    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        windows.append(values[start:end])

    if not windows:
        raise RuntimeError("No windows generated — check window_size/stride")

    X = np.stack(windows)

    print(f"[INFO] Generated windows: {X.shape}")
    print(f"[INFO] Window shape: ({window_size}, {n_features})")
    print(f"[INFO] dtype: {X.dtype}")

    if np.isnan(X).any():
        raise ValueError("NaNs detected in windowed data")

    # -----------------------------
    # STEP 1: Compute scaler stats
    # -----------------------------
    flat = X.reshape(-1, n_features)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)

    # production safety: avoid division by zero
    std_safe = np.where(std < 1e-12, 1.0, std)

    # -----------------------------
    # STEP 2: Apply normalization
    # -----------------------------
    X_norm = (X - mean) / std_safe

    if np.isnan(X_norm).any() or np.isinf(X_norm).any():
        raise ValueError("NaN/Inf detected after normalization")

    # Quick normalization check (optional but reassuring)
    flat_norm = X_norm.reshape(-1, n_features)
    mean_norm = flat_norm.mean(axis=0)
    std_norm = flat_norm.std(axis=0)
    print("[INFO] Normalization check (should be ~0 mean, ~1 std):")
    for name, m, s in zip(feature_cols, mean_norm, std_norm):
        print(f"  {name:>22s}  mean={m:.6f}  std={s:.6f}")

    # -----------------------------
    # STEP 3: Save artifacts
    # -----------------------------
    out_npz = paths.processed_dir / "train_windows_norm.npz"
    out_scaler = paths.processed_dir / "windowing_scaler.json"

    np.savez_compressed(
        out_npz,
        X=X_norm.astype(np.float32),  # float32 is enough + smaller files
        feature_cols=np.array(feature_cols, dtype=object),
        window_size=np.array(window_size),
        stride=np.array(stride),
    )

    scaler_payload = {
        "source_parquet": str(parquet_path).replace("\\", "/"),
        "timestamp_col": ts_col,
        "features": feature_cols,
        "window_size": int(window_size),
        "stride": int(stride),
        "n_windows": int(X_norm.shape[0]),
        "window_shape": [int(window_size), int(n_features)],
        "X_shape": [int(x) for x in X_norm.shape],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "std_safe": std_safe.tolist(),
        "dtype_saved": "float32",
    }

    out_scaler.write_text(json.dumps(scaler_payload, indent=2), encoding="utf-8")

    print(f"[SUCCESS] Saved windows: {out_npz}")
    print(f"[SUCCESS] Saved scaler+windowing config: {out_scaler}")


if __name__ == "__main__":
    main()
