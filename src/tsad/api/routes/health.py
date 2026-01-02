from fastapi import APIRouter
from tsad.api.core.settings import settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.version,
        "data_root": settings.data_root,
        "reports_dir": settings.reports_dir,
    }
