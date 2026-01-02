from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, conint, confloat


class DecideManyRequest(BaseModel):
    rel_paths: list[str] = Field(
        ...,
        min_length=1,
        description="List of SKAB relative CSV paths (e.g. valve1/1.csv)",
    )
    warning_k: conint(ge=1) = 3
    critical_k: conint(ge=1) = 5
    ood_margin: confloat(ge=0) = 0.0

    # Optional override, keep default stable
    config_path: str = "configs/default.yaml"


class DecideManyError(BaseModel):
    file: str = Field(..., description="Input rel_path that failed")
    error: str = Field(..., description="Exception class name")
    message: str = Field(..., description="Human-readable error message")


class DecideManySummary(BaseModel):
    n_total: int
    n_scored: int
    n_errors: int
    n_ood: int
    n_warning: int
    n_critical: int
    n_ok: int


class DecideManyResponse(BaseModel):
    # Keep flexible because decide_skab_file returns a rich dict
    results: list[dict[str, Any]]

    # Typed summary for dashboards
    summary: DecideManySummary

    # Optional failures (batch should not hard-fail)
    errors: Optional[list[DecideManyError]] = None
