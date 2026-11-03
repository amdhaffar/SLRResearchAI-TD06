import json

import pytest
from pydantic import ValidationError

from slrresearch.handoff import EXPECTED_TOOLS, load_search_plan


def test_handoff_charge_reellement_le_json_fourni(tmp_path):
    payload = {
        "schema_version": "1.0",
        "origin": "td02_runtime_export",
        "produced_by": "TD02",
        "topic": "Sujet transmis par le TD02",
        "questions": [{"identifier": "RQ1", "question": "Question assez longue ?"}],
        "queries": {name: f"requête-{name}" for name in EXPECTED_TOOLS},
    }
    path = tmp_path / "search_plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    artifact = load_search_plan(path)
    assert artifact.topic == payload["topic"]
    assert artifact.queries == payload["queries"]


def test_handoff_refuse_un_moteur_manquant(tmp_path):
    payload = {
        "origin": "test",
        "produced_by": "TD02",
        "topic": "Sujet transmis",
        "questions": [{"identifier": "RQ1", "question": "Question assez longue ?"}],
        "queries": {"openalex": "requête"},
    }
    path = tmp_path / "search_plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_search_plan(path)


def test_handoff_refuse_un_fichier_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_search_plan(tmp_path / "absent.json")
