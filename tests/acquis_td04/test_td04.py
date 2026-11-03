from pathlib import Path

import pytest
from pydantic import ValidationError

from slrresearch.artifacts import load_search_artifact
from slrresearch.screening_agent import (
    ScreeningAgent,
    ScreeningIndicators,
    apply_screening_policy,
    choose_preliminary_verdict,
    normalize_paper_text,
    observe_indicators,
    requires_human_review,
)


ARTIFACT = Path(__file__).resolve().parent / "fixtures" / "search_results.json"


def test_acquis_handoff_td03_charge_un_artefact_valide():
    artifact = load_search_artifact(ARTIFACT)
    assert artifact.origin == "TD03"
    assert artifact.pedagogical_fixture is True
    assert len(artifact.papers) == artifact.search.unique_results == 5


def test_acquis_handoff_refuse_une_mauvaise_origine(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        ARTIFACT.read_text(encoding="utf-8").replace('"TD03"', '"TD99"', 1),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_search_artifact(invalid)


def test_todo_1_normalise_titre_et_resume():
    paper = load_search_artifact(ARTIFACT).papers[0]
    assert normalize_paper_text(paper) == (
        f"{paper.title} {paper.abstract}".lower()
    )


def test_todo_2_detecte_conflit_et_information_insuffisante():
    papers = load_search_artifact(ARTIFACT).papers
    conflict = observe_indicators(papers[3], {"slr"}, {"medicine"})
    missing = observe_indicators(papers[4], {"slr"}, {"medicine"})
    assert conflict.conflict is True
    assert missing.information_sufficient is False


def test_todo_3_priorite_exclusion_et_raisons():
    indicators = ScreeningIndicators(["slr"], ["medicine"], True, True)
    decision, confidence, reasons = choose_preliminary_verdict(indicators)
    assert decision == "exclude"
    assert confidence == 0.95
    assert reasons


def test_todo_4_transfere_abstention_conflit_et_information_incomplete():
    cases = [
        ScreeningIndicators([], [], True, False),
        ScreeningIndicators(["slr"], ["medicine"], True, True),
        ScreeningIndicators([], [], False, False),
    ]
    provisional = [
        ("uncertain", 0.45),
        ("exclude", 0.95),
        ("uncertain", 0.45),
    ]
    assert all(
        requires_human_review(decision, confidence, case)
        for (decision, confidence), case in zip(provisional, cases)
    )


def test_todo_5_verdict_conserve_les_indicateurs():
    indicators = ScreeningIndicators(["slr"], [], True, False)
    decision = apply_screening_policy("P01", indicators)
    assert decision.indicators is indicators
    assert decision.needs_human_review is False


def test_todo_6_agent_consomme_directement_les_papers_du_td03():
    artifact = load_search_artifact(ARTIFACT)
    result = ScreeningAgent().run(
        artifact.papers,
        {"slr", "screening", "systematic review"},
        {"medicine", "clinical trial"},
    )
    assert len(result["decisions"]) == len(artifact.papers)
    assert {item.paper_id for item in result["human_queue"]} == {
        "P02",
        "P04",
        "P05",
    }
