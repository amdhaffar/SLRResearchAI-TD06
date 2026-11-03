"""Contrat du handoff TD02 → TD03Bis (code d'infrastructure fourni)."""

import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parent.parent / "artifacts" / "search_plan.json"
)
EXPECTED_TOOLS = {"openalex", "crossref", "semantic_scholar"}


class ResearchQuestionArtifact(BaseModel):
    identifier: str
    question: str = Field(min_length=10)


class SearchPlanArtifact(BaseModel):
    schema_version: str = "1.0"
    origin: str
    produced_by: str
    topic: str = Field(min_length=3)
    questions: list[ResearchQuestionArtifact] = Field(min_length=1)
    queries: dict[str, str]

    @model_validator(mode="after")
    def validate_queries(self):
        if set(self.queries) != EXPECTED_TOOLS:
            raise ValueError(
                "queries doit contenir exactement openalex, crossref et "
                "semantic_scholar"
            )
        if any(not query.strip() for query in self.queries.values()):
            raise ValueError("Chaque moteur doit recevoir une requête non vide")
        return self


def load_search_plan(path: Path = DEFAULT_ARTIFACT_PATH) -> SearchPlanArtifact:
    """Charge et valide l'artefact réellement consommé par le TD03Bis."""
    if not path.exists():
        raise FileNotFoundError(
            f"Artefact TD02 absent : {path}. Exécutez le TD02 ou restaurez "
            "le fichier pédagogique fourni."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SearchPlanArtifact.model_validate(payload)
