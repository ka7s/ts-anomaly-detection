from fastapi import APIRouter, HTTPException

from tsad.api.schemas.decide import DecideRequest
from tsad.inference.decision import decide_skab_file

router = APIRouter(prefix="/decide", tags=["decide"])


@router.post("")
def decide(payload: DecideRequest):
    try:
        # 👇 THIS is where the line you asked about goes
        return decide_skab_file(
            rel_path=payload.rel_path,
            warning_k=payload.warning_k,
            critical_k=payload.critical_k,
            ood_margin=payload.ood_margin,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
