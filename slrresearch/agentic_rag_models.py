from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class EvidenceFacet(BaseModel):
    """Aspect attendu dans les passages, utilisé pour mesurer la couverture."""

    id: str
    label: str
    keywords: list[str] = Field(min_length=1)
    query_hint: str


class RetrievalAction(BaseModel):
    """Décision explicite et validée prise par l'agent."""

    action: Literal["retrieve", "finish", "abstain"]
    tool: Literal["lexical", "semantic"] | None = None
    query: str = ""
    reason: str

    @model_validator(mode="after")
    def coherent_action(self):
        if self.action == "retrieve" and (self.tool is None or not self.query.strip()):
            raise ValueError("Une action retrieve exige un outil et une requête")
        if self.action != "retrieve" and self.tool is not None:
            raise ValueError("Une action terminale ne doit pas appeler un outil")
        return self


class AgentTraceEvent(BaseModel):
    iteration: int = Field(ge=1)
    action: str
    tool: str | None = None
    query: str = ""
    found: int = Field(ge=0)
    covered_facets: list[str] = Field(default_factory=list)
    missing_facets: list[str] = Field(default_factory=list)
    reason: str


class AdaptiveRAGState(BaseModel):
    question: str
    facets: list[EvidenceFacet]
    max_iterations: int = Field(default=2, ge=1, le=5)
    iteration: int = 0
    retrieval_calls: int = 0
    selected_chunks: list[dict] = Field(default_factory=list)
    covered_facets: list[str] = Field(default_factory=list)
    missing_facets: list[str] = Field(default_factory=list)
    trace: list[AgentTraceEvent] = Field(default_factory=list)
    status: Literal["running", "completed", "abstained", "budget_exhausted"] = "running"


class AgenticCase(BaseModel):
    id: str
    label: str
    question: str
    facets: list[EvidenceFacet]
    relevant_chunk_ids: list[str]
    baseline_tool: Literal["lexical", "semantic"] = "lexical"
    baseline_k: int = Field(default=1, ge=1)
    pedagogical_goal: str = ""
    expected_behavior: str = ""


class ComparisonReport(BaseModel):
    case_id: str
    baseline: dict
    agentic: dict
    delta_recall: float
    extra_retrieval_calls: int
