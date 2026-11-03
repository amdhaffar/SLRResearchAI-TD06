from typing import Literal
import re
import unicodedata

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from slrresearch.artifacts import PaperRecord
from slrresearch.screening_agent import ScreeningDecision


class LLMScreeningDecision(BaseModel):
    paper_id: str
    decision: Literal["include", "exclude", "uncertain"]
    confidence: float = Field(ge=0, le=1)
    reasons: list[str] = Field(min_length=1)
    evidence: str = Field(
        description=(
            "Courte preuve copiée du titre ou du résumé ; son ancrage lexical "
            "sera contrôlé localement"
        )
    )
    needs_human_review: bool

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value):
        """Accepte 0..1 ou un pourcentage 0..100, puis stocke 0..1."""
        number = float(value)
        if 1 < number <= 100:
            return number / 100
        return number


class ComparisonResult(BaseModel):
    paper_id: str
    deterministic_decision: Literal["include", "exclude", "uncertain"]
    llm_decision: Literal["include", "exclude", "uncertain"]
    agreement: bool
    needs_human_review: bool
    escalation_reason: str


def normalize_for_matching(text: str) -> str:
    """Neutralise les variations sans modifier le sens lexical."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[^\w%]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def build_screening_messages(
    paper: PaperRecord,
    topic: str,
    inclusion_terms: set[str],
    exclusion_terms: set[str],
) -> list:
    # ACQUIS TD05 — Construire les messages système et utilisateur.
    system_prompt = (
        "Tu es un évaluateur méthodologique de SLR. Décide uniquement entre "
        "include, exclude et uncertain à partir du titre et du résumé fournis. "
        "N'invente aucune information. confidence doit être un nombre entre 0 et 1. "
        "evidence doit être une courte citation "
        "copiée exactement du titre ou du résumé. Si l'information manque, choisis "
        "uncertain, utilise une preuve vide et demande une validation humaine."
    )
    user_prompt = (
        f"Sujet de la SLR : {topic}\n"
        f"Critères d'inclusion : {', '.join(sorted(inclusion_terms))}\n"
        f"Critères d'exclusion : {', '.join(sorted(exclusion_terms))}\n"
        f"Identifiant technique à recopier exactement : {paper.id}\n"
        f"Titre : {paper.title}\n"
        f"Résumé : {paper.abstract or '[ABSENT]'}"
    )
    return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]


def invoke_structured_screening(
    model,
    paper: PaperRecord,
    topic: str,
    inclusion_terms: set[str],
    exclusion_terms: set[str],
) -> LLMScreeningDecision:
    # ACQUIS TD05 — Effectuer un seul appel avec sortie Pydantic.
    messages = build_screening_messages(paper, topic, inclusion_terms, exclusion_terms)
    structured_model = model.with_structured_output(LLMScreeningDecision)
    decision = structured_model.invoke(messages)
    # L'identifiant relève du contrôle de l'application, jamais du jugement du LLM.
    return decision.model_copy(update={"paper_id": paper.id})


def validate_grounding(paper: PaperRecord, decision: LLMScreeningDecision) -> None:
    """Contrôle l'ancrage de la preuve, pas la validité scientifique du raisonnement."""
    # ACQUIS TD05 — Accepter l'abstention vide ou une preuve ancrée.
    source_text = normalize_for_matching(f"{paper.title} {paper.abstract}")
    evidence = decision.evidence.strip()
    normalized_evidence = normalize_for_matching(evidence)
    valid_empty_evidence = not evidence and decision.decision == "uncertain"
    evidence_found = bool(evidence) and normalized_evidence in source_text
    if not (valid_empty_evidence or evidence_found):
        raise ValueError("Preuve LLM absente du titre et du résumé")


def screen_with_llm(
    model,
    paper: PaperRecord,
    topic: str,
    inclusion_terms: set[str],
    exclusion_terms: set[str],
) -> LLMScreeningDecision:
    decision = invoke_structured_screening(
        model, paper, topic, inclusion_terms, exclusion_terms
    )
    try:
        validate_grounding(paper, decision)
    except ValueError as exc:
        decision = decision.model_copy(
            update={
                "decision": "uncertain",
                "confidence": 0.0,
                "reasons": [*decision.reasons, f"Contrôle local : {exc}"],
                "evidence": "",
                "needs_human_review": True,
            }
        )
    return decision


def compare_decisions(
    deterministic: ScreeningDecision,
    llm: LLMScreeningDecision,
) -> ComparisonResult:
    # ACQUIS TD05 — Comparer les sorties et décider AUTO ou HUMAN.
    agreement = deterministic.decision == llm.decision
    # ACQUIS TD05a — Lire la demande humaine de la version A.
    deterministic_requests_human = deterministic.needs_human_review
    # ACQUIS TD05b — Lire la demande humaine de la version B.
    llm_requests_human = llm.needs_human_review
    needs_human_review = (
        deterministic_requests_human
        or llm_requests_human
        or not agreement
    )
    if not agreement:
        reason = "Désaccord entre la règle déterministe et l'analyse sémantique"
    elif deterministic.needs_human_review or llm.needs_human_review:
        reason = "Au moins un agent demande une validation humaine"
    else:
        reason = "Accord exploitable automatiquement"
    return ComparisonResult(
        paper_id=deterministic.paper_id,
        deterministic_decision=deterministic.decision,
        llm_decision=llm.decision,
        agreement=agreement,
        needs_human_review=needs_human_review,
        escalation_reason=reason,
    )
