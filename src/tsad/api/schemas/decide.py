from pydantic import BaseModel, Field


class DecideRequest(BaseModel):
    rel_path: str = Field(..., description="Relative path under your data root, e.g. 'valve1/1.csv'")
    warning_k: int = Field(3, ge=1, description="Warning multiplier (k)")
    critical_k: int = Field(5, ge=1, description="Critical multiplier (k)")
    ood_margin: float = Field(0.0, ge=0.0, description="Extra margin added to the OOD threshold")


class DecideResponse(BaseModel):
    # Placeholder response contract for now (Step 3 will return the real decision JSON)
    detail: str
