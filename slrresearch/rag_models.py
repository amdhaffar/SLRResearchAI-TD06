from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    paper_id: str
    chunk_id: str
    page: int = Field(ge=1)
    text: str = Field(min_length=1)


class ProposedEvidence(BaseModel):
    answer: str
    supporting_quote: str
    paper_id: str
    chunk_id: str
    page: int = Field(ge=1)


class VerifiedEvidence(ProposedEvidence):
    status: Literal["verified", "human_review", "rejected"]
    verification_reason: str

