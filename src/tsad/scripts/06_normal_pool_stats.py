from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsad.config import load_config, get_paths
from tsad.data.download import download_skab_repo_zip
from tsad.data.load import load_one_csv


def count_strict_normal_windows(labels: np.ndarray, window_size: int, stride: int) -> int:
    """
    Strict normal window = ALL labels in the window are 0.
    """
    n = len(labels)
    count = 0
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        if labels[start:end].max() == 0:
            count += 1
    return count


def main() -> None:
    cfg = load_config("configs/default.yaml")
    paths = get_paths(cfg)

    manifest_path = paths.processed_dir / "skab_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files_with_anomaly = manifest["files_with_anomaly"]

    ts_col = cfg["dataset_prep"]["timestamp_col"]
    anomaly_col = cfg["dataset_prep"]["anomaly_col"]

    # Use same window params as training
    scaler_path = paths.processed_dir / "windowing_scaler.json"
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    window_size = int(scaler["window_size"])
    stride = int(scaler["stride"])

    extracted_repo_dir = download_skab_repo_zip(cfg["skab"]["repo_zip_url"], paths.raw_dir)
    base = (
        extracted_repo_dir
        / cfg["skab"]["extracted_root_dirname"]
        / cfg["skab"]["data_subdir"]
    )

    rows = []
    for rel in files_with_anomaly:
        csv_path = base / rel
        df = load_one_csv(csv_path)

        # basic clean on ts
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

        if anomaly_col not in df.columns:
            continue

        labels = pd.to_numeric(df[anomaly_col], errors="coerce").fillna(0).to_numpy(dtype=int)
        n_rows = len(labels)
        anomaly_rate = float(labels.mean())

        n_strict = count_strict_normal_windows(labels, window_size=window_size, stride=stride)

        rows.append(
            {
                "file": rel,
                "rows": n_rows,
                "anomaly_rate": anomaly_rate,
                "strict_normal_windows": n_strict,
            }
        )

    out = pd.DataFrame(rows).sort_values(["strict_normal_windows", "anomaly_rate"], ascending=[False, True])

    print("[INFO] Top files by strict_normal_windows:")
    print(out.head(10).to_string(index=False))

    print(f"[INFO] Total labeled files: {len(files_with_anomaly)}")
    print(f"[INFO] Total strict normal windows across labeled files: {int(out['strict_normal_windows'].sum())}")

    print("[SUCCESS] Step 1 complete: normal-pool statistics computed (no training yet).")


if __name__ == "__main__":
    main()
