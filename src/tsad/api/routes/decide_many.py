from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException

from tsad.api.core.settings import settings
from tsad.api.schemas.decide_many import (
    DecideManyRequest,
    DecideManyResponse,
    DecideManyError,
    DecideManySummary,
)
from tsad.inference.decision import decide_skab_file

logger = logging.getLogger("tsad.api")

router = APIRouter(tags=["decide"])


@router.post("/decide_many", response_model=DecideManyResponse)
def decide_many(payload: DecideManyRequest) -> DecideManyResponse:
    # -------- Validation --------
    if payload.critical_k < payload.warning_k:
        raise HTTPException(
            status_code=400,
            detail="critical_k should be >= warning_k",
        )

    if len(payload.rel_paths) > settings.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files. max_batch_size={settings.max_batch_size}",
        )

    logger.info(
        f"/decide_many n={len(payload.rel_paths)} warning_k={payload.warning_k} "
        f"critical_k={payload.critical_k} ood_margin={payload.ood_margin}"
    )

    # -------- Processing --------
    results: list[dict] = []
    errors: list[DecideManyError] = []

    for rel_path in payload.rel_paths:
        try:
            out = decide_skab_file(
                rel_path=rel_path,
                warning_k=payload.warning_k,
                critical_k=payload.critical_k,
                ood_margin=payload.ood_margin,
                config_path=payload.config_path,
            )
            results.append(out)

        except Exception as e:
            # One bad file must NOT kill the batch
            errors.append(
                DecideManyError(
                    file=rel_path,
                    error=type(e).__name__,
                    message=str(e),
                )
            )

    # -------- Summary --------
    summary = DecideManySummary(
        n_total=len(payload.rel_paths),
        n_scored=len(results),
        n_errors=len(errors),
        n_ood=sum(1 for r in results if r.get("ood") is True),
        n_warning=sum(1 for r in results if r.get("warning") is True),
        n_critical=sum(1 for r in results if r.get("critical") is True),
        n_ok=sum(
            1
            for r in results
            if (r.get("ood") is False and r.get("warning") is False and r.get("critical") is False)
        ),
    )

    # -------- Response --------
    return DecideManyResponse(
        results=results,
        summary=summary,
        errors=errors if errors else None,
    )
