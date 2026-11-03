import pytest

from slrresearch.artifacts import PaperRecord
from slrresearch.llm_screening_agent import (
    LLMScreeningDecision,
    build_screening_messages,
    compare_decisions,
    invoke_structured_screening,
    screen_with_llm,
    validate_grounding,
)
from slrresearch.screening_agent import ScreeningDecision, ScreeningIndicators


PAPER = PaperRecord(
    id="P42",
    title="LLM-assisted screening",
    abstract="The approach supports systematic review study selection.",
    source="test",
    url="https://example.org/p42",
    open_access=True,
)


class FakeStructuredModel:
    def __init__(self, decision):
        self.decision = decision
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        assert len(messages) == 2
        return self.decision


class FakeModel:
    def __init__(self, decision):
        self.structured = FakeStructuredModel(decision)
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self.structured


def llm_decision(**changes):
    data = {
        "paper_id": "P42",
        "decision": "include",
        "confidence": 0.88,
        "reasons": ["Le résumé décrit la sélection des études"],
        "evidence": "systematic review study selection",
        "needs_human_review": False,
    }
    data.update(changes)
    return LLMScreeningDecision(**data)


def deterministic(decision="include", human=False):
    return ScreeningDecision(
        paper_id="P42",
        decision=decision,
        confidence=0.9,
        reasons=["règle de test"],
        indicators=ScreeningIndicators(
            inclusion_matches=["screening"],
            exclusion_matches=[],
            information_sufficient=True,
            conflict=False,
        ),
        needs_human_review=human,
    )


def test_todo_7_prompt_contient_contexte_contraintes_et_papier():
    messages = build_screening_messages(PAPER, "SLR et LLM", {"screening"}, {"medicine"})
    joined = "\n".join(str(message.content) for message in messages).lower()
    assert len(messages) == 2
    assert "include" in joined and "exclude" in joined and "uncertain" in joined
    assert "n'invente" in joined or "invent" in joined
    assert "slr et llm" in joined and PAPER.id.lower() in joined
    assert PAPER.title.lower() in joined and PAPER.abstract.lower() in joined


def test_todo_8_un_seul_appel_structure_et_identifiant_controle():
    model = FakeModel(llm_decision(paper_id="DOI-retourne-par-le-LLM"))
    result = invoke_structured_screening(
        model, PAPER, "SLR", {"screening"}, {"medicine"}
    )
    assert result.decision == "include"
    assert result.paper_id == PAPER.id
    assert model.schema is LLMScreeningDecision
    assert model.structured.calls == 1


def test_todo_9_preuve_inventee_refusee_et_abstention_vide_acceptee():
    with pytest.raises(ValueError, match="Preuve"):
        validate_grounding(PAPER, llm_decision(evidence="citation inventée"))
    validate_grounding(
        PAPER,
        llm_decision(decision="uncertain", evidence="", needs_human_review=True),
    )


def test_preuve_inverifiable_degrade_le_resultat_sans_interrompre_le_lot():
    model = FakeModel(llm_decision(evidence="citation inventée"))
    result = screen_with_llm(model, PAPER, "SLR", {"screening"}, {"medicine"})
    assert result.decision == "uncertain"
    assert result.confidence == 0.0
    assert result.needs_human_review is True
    assert result.evidence == ""


def test_todo_10_desaccord_ou_demande_declenche_arbitrage_humain():
    # Cas 1 — Les verdicts divergent : C doit demander l'humain.
    disagreement = compare_decisions(deterministic("exclude"), llm_decision())
    assert disagreement.agreement is False
    assert disagreement.needs_human_review is True

    # Cas 2 — A et B sont d'accord et aucun ne demande l'humain : AUTO.
    agreement = compare_decisions(deterministic(), llm_decision())
    assert agreement.agreement is True
    assert agreement.needs_human_review is False

    # Cas 3 — Accord, mais la version A demande explicitement l'humain.
    requested_by_a = compare_decisions(
        deterministic(human=True),
        llm_decision(),
    )
    assert requested_by_a.agreement is True
    assert requested_by_a.needs_human_review is True

    # Cas 4 — Accord, mais la version B demande explicitement l'humain.
    requested_by_b = compare_decisions(
        deterministic(),
        llm_decision(needs_human_review=True),
    )
    assert requested_by_b.agreement is True
    assert requested_by_b.needs_human_review is True


def test_contrat_normalise_une_confiance_exprimee_en_pourcentage():
    assert llm_decision(confidence=100).confidence == 1.0
    assert llm_decision(confidence=82).confidence == 0.82
