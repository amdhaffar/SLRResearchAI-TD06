import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .rag_models import VerifiedEvidence


class PaperRecord(BaseModel):
    id: str
    title: str
    abstract: str = ""
    doi: str = ""
    source: str
    url: str = ""
    open_access: bool = False


class SearchArtifact(BaseModel):
    schema_version: Literal["1.0"]
    origin: Literal["TD03"]
    topic: str
    papers: list[PaperRecord]


class ScreeningRecord(BaseModel):
    paper_id: str
    final_decision: Literal["include", "exclude"]
    decided_by: Literal["automatic", "human"]
    reason: str


class ScreeningArtifact(BaseModel):
    schema_version: Literal["1.0"]
    origin: Literal["TD04"]
    source_artifact: str
    decisions: list[ScreeningRecord]


class DocumentRecord(BaseModel):
    paper_id: str
    status: Literal["downloaded", "manual_required", "unavailable"]
    source_url: str
    extracted_pages_key: str | None = None


class DocumentManifest(BaseModel):
    schema_version: Literal["1.0"]
    origin: Literal["DOCUMENT_PREPARATION"]
    source_artifact: str
    pedagogical_fixture: bool = True
    documents: list[DocumentRecord]


class SourceDocument(BaseModel):
    paper_id: str
    title: str
    pages: list[str] = Field(min_length=1)


def load_rag_handoff(artifact_dir: Path) -> tuple[list[SourceDocument], dict]:
    """Charge et joint les artefacts TD03 → TD04 → préparation documentaire."""
    search = SearchArtifact.model_validate_json(
        (artifact_dir / "search_results.json").read_text(encoding="utf-8")
    )
    screening = ScreeningArtifact.model_validate_json(
        (artifact_dir / "screening_decisions.json").read_text(encoding="utf-8")
    )
    manifest = DocumentManifest.model_validate_json(
        (artifact_dir / "document_manifest.json").read_text(encoding="utf-8")
    )
    extracted = json.loads(
        (artifact_dir / "extracted_pages.json").read_text(encoding="utf-8")
    )

    papers = {paper.id: paper for paper in search.papers}
    included = {
        decision.paper_id
        for decision in screening.decisions
        if decision.final_decision == "include"
    }
    available = {
        document.paper_id: document
        for document in manifest.documents
        if document.status == "downloaded" and document.extracted_pages_key
    }
    selected_ids = included & set(available) & set(papers)
    documents = [
        SourceDocument(
            paper_id=paper_id,
            title=papers[paper_id].title,
            pages=extracted[available[paper_id].extracted_pages_key],
        )
        for paper_id in sorted(selected_ids)
    ]
    lineage = {
        "topic": search.topic,
        "found": len(search.papers),
        "included": len(included),
        "documents_available": len(available),
        "rag_documents": len(documents),
        "excluded_from_rag": sorted(set(papers) - selected_ids),
        "sources": {
            "search": "TD03/search_results.json",
            "screening": "TD04/screening_decisions.json",
            "documents": "document_manifest.json",
        },
    }
    return documents, lineage


def save_evidence_artifact(evidence: VerifiedEvidence, path: Path) -> None:
    """Produit l'artefact consommable par le prochain TD de synthèse."""
    payload = {
        "schema_version": "1.0",
        "origin": "RAG_TD05",
        "source_artifacts": [
            "search_results.json",
            "screening_decisions.json",
            "document_manifest.json",
        ],
        "evidence": [evidence.model_dump()],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
