import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class PaperRecord(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    abstract: str = ""
    doi: str = ""
    source: str = Field(min_length=1)
    url: str = ""
    open_access: bool = False


class SearchMetadata(BaseModel):
    queries: dict[str, str]
    sources_attempted: list[str]
    stop_reason: str
    raw_results: int = Field(ge=0)
    unique_results: int = Field(ge=0)


class SearchArtifact(BaseModel):
    schema_version: Literal["1.0"]
    origin: Literal["TD03"]
    topic: str = Field(min_length=1)
    created_at: str
    pedagogical_fixture: bool = False
    note: str = ""
    search: SearchMetadata
    papers: list[PaperRecord] = Field(min_length=1)


def load_search_artifact(path: Path) -> SearchArtifact:
    """Charge et valide le contrat de transmission TD03 → TD04."""
    if not path.exists():
        raise FileNotFoundError(f"Artefact TD03 introuvable : {path}")
    return SearchArtifact.model_validate_json(path.read_text(encoding="utf-8"))
