from pydantic import BaseModel, Field
from typing import Tuple

class TraceRequest(BaseModel):
    base_prompt: str = Field(..., description="The context prompt (e.g., 'The capital of')")
    claim_prompt: str = Field(..., description="The full claim (e.g., 'The capital of France is')")
    top_k: int = Field(default=50, ge=1, le=100)

class TraceResponse(BaseModel):
    base_entropy: float
    claim_entropy: float
    expert_location: Tuple[int, int]
    risk_score: float
    is_hallucination: bool