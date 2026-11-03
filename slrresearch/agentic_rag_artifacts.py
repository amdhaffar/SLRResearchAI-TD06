from __future__ import annotations

import json
from pathlib import Path

from .agentic_rag_models import AgenticCase, ComparisonReport


def load_agentic_cases(path: Path) -> list[AgenticCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("origin") != "TD06_PROTOCOL":
        raise ValueError("Artefact de protocole TD06 invalide")
    return [AgenticCase.model_validate(item) for item in payload["cases"]]


def save_comparison_artifact(
    report: ComparisonReport, trace: list[dict], path: Path
) -> None:
    payload = {
        "schema_version": "1.0",
        "origin": "TD06_AGENTIC_RAG",
        "source_artifact": "agentic_rag_cases.json",
        "comparison": report.model_dump(),
        "trace": trace,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
