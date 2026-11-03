import json
from pathlib import Path

from slrresearch.rag_artifacts import load_rag_handoff, save_evidence_artifact
from slrresearch.rag_models import VerifiedEvidence


ARTIFACTS = Path(__file__).parents[2] / "artifacts"


def test_handoff_ne_garde_que_les_papiers_inclus_et_disponibles():
    documents, lineage = load_rag_handoff(ARTIFACTS)
    assert {item.paper_id for item in documents} == {"W4410000915", "W4402635906"}
    assert "W4410813691" in lineage["excluded_from_rag"]


def test_preuve_devient_un_artefact_pour_le_td_suivant(tmp_path):
    evidence = VerifiedEvidence(
        answer="résultat", supporting_quote="citation", paper_id="P1",
        chunk_id="P1_p1_c1", page=1, status="verified",
        verification_reason="contrôle réussi",
    )
    target = tmp_path / "verified_evidence.json"
    save_evidence_artifact(evidence, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["origin"] == "RAG_TD05"
    assert payload["evidence"][0]["chunk_id"] == "P1_p1_c1"
