from __future__ import annotations

import json

from tsad.config import load_config, get_paths
from tsad.data.download import download_skab_repo_zip
from tsad.data.load import find_skab_csv_files, load_one_csv
from tsad.data.validate import validate_schema


def main() -> None:
    cfg = load_config("configs/default.yaml")
    paths = get_paths(cfg)

    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Downloading SKAB repository...")
    extracted_repo_dir = download_skab_repo_zip(
        cfg["skab"]["repo_zip_url"],
        paths.raw_dir,
    )

    print("[INFO] Searching for CSV files...")
    csv_files = find_skab_csv_files(
        extracted_repo_dir=extracted_repo_dir,
        extracted_root_dirname=cfg["skab"]["extracted_root_dirname"],
        data_subdir=cfg["skab"]["data_subdir"],
    )
    print(f"[INFO] Found {len(csv_files)} CSV files")

    timestamp_col = cfg["dataset_prep"]["timestamp_col"]
    anomaly_col = cfg["dataset_prep"]["anomaly_col"]

    has_anomaly: list = []
    no_anomaly: list = []

    for p in csv_files:
        df = load_one_csv(p)
        validate_schema(df, timestamp_col=timestamp_col, anomaly_col=anomaly_col)

        if anomaly_col in df.columns:
            has_anomaly.append(p)
        else:
            no_anomaly.append(p)

    print(f"[SUCCESS] Validated {len(csv_files)} files")
    print(f"[INFO] Files WITH anomaly column: {len(has_anomaly)}")
    print(f"[INFO] Files WITHOUT anomaly column: {len(no_anomaly)}")
    print("[INFO] Example WITH anomaly:", has_anomaly[0].name if has_anomaly else "NONE")
    print("[INFO] Example WITHOUT anomaly:", no_anomaly[0].name if no_anomaly else "NONE")

    # --- build dataset manifest with unique relative paths ---
    base = (
        extracted_repo_dir
        / cfg["skab"]["extracted_root_dirname"]
        / cfg["skab"]["data_subdir"]
    )

    manifest = {
        "total_files": len(csv_files),
        "files_with_anomaly": [
            str(p.relative_to(base)).replace("\\", "/") for p in has_anomaly
        ],
        "files_without_anomaly": [
            str(p.relative_to(base)).replace("\\", "/") for p in no_anomaly
        ],
    }

    manifest_path = paths.processed_dir / "skab_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] Saved dataset manifest to {manifest_path}")


if __name__ == "__main__":
    main()
