import os
from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "tsad-inference"
    version: str = "0.1.0"

    # Configurable via env vars
    data_root: str = os.getenv("TSAD_DATA_ROOT", r"data\raw\skab_repo\SKAB-master\data")
    reports_dir: str = os.getenv("TSAD_REPORTS_DIR", r"reports")

    # Hardening: max batch size for /decide_many
    max_batch_size: int = int(os.getenv("TSAD_MAX_BATCH_SIZE", "200"))


settings = Settings()
