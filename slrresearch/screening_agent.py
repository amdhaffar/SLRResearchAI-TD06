from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ScreeningIndicators:
    inclusion_matches: list[str]
    exclusion_matches: list[str]
    information_sufficient: bool
    conflict: bool


@dataclass(frozen=True)
class ScreeningDecision:
    paper_id: str
    decision: Literal["include", "exclude", "uncertain"]
    confidence: float
    reasons: list[str]
    indicators: ScreeningIndicators
    needs_human_review: bool


def normalize_paper_text(paper) -> str:
    # ACQUIS TD05 — Normaliser le titre et le résumé.
    return f"{paper.title} {paper.abstract}".lower()


def observe_indicators(paper, include_terms, exclude_terms):
    text = normalize_paper_text(paper)
    # ACQUIS TD05 — Extraire les signaux d'inclusion et d'exclusion.
    inclusion_matches = sorted(
        term for term in include_terms if term.lower() in text
    )
    exclusion_matches = sorted(
        term for term in exclude_terms if term.lower() in text
    )
    return ScreeningIndicators(
        inclusion_matches=inclusion_matches,
        exclusion_matches=exclusion_matches,
        information_sufficient=bool(paper.title.strip() and paper.abstract.strip()),
        conflict=bool(inclusion_matches and exclusion_matches),
    )


def choose_preliminary_verdict(
    indicators: ScreeningIndicators,
) -> tuple[Literal["include", "exclude", "uncertain"], float, list[str]]:
    # ACQUIS TD05 — Appliquer la priorité exclusion > inclusion > abstention.
    if indicators.exclusion_matches:
        decision = "exclude"
        confidence = 0.95
        reasons = [
            f"Critère d'exclusion détecté : {term}"
            for term in indicators.exclusion_matches
        ]
    elif indicators.inclusion_matches:
        decision = "include"
        confidence = 0.90
        reasons = [
            f"Critère d'inclusion détecté : {term}"
            for term in indicators.inclusion_matches
        ]
    else:
        decision = "uncertain"
        confidence = 0.45
        reasons = ["Aucun signal lexical ne permet de trancher"]

    return decision, confidence, reasons


def requires_human_review(
    decision: str,
    confidence: float,
    indicators: ScreeningIndicators,
    human_threshold: float = 0.80,
) -> bool:
    # ACQUIS TD05 — Déterminer si le verdict peut être automatisé.
    return (
        decision == "uncertain"
        or confidence < human_threshold
        or indicators.conflict
        or not indicators.information_sufficient
    )


def apply_screening_policy(
    paper_id: str,
    indicators: ScreeningIndicators,
    human_threshold: float = 0.80,
) -> ScreeningDecision:
    # ACQUIS TD05 — Construire le verdict structuré et justifiable.
    decision, confidence, reasons = choose_preliminary_verdict(indicators)

    if not indicators.information_sufficient:
        reasons.append("Titre ou résumé insuffisant")
    if indicators.conflict:
        reasons.append("Conflit entre critères d'inclusion et d'exclusion")

    needs_human_review = requires_human_review(
        decision, confidence, indicators, human_threshold
    )
    return ScreeningDecision(
        paper_id,
        decision,
        confidence,
        reasons,
        indicators,
        needs_human_review,
    )


class ScreeningAgent:
    def run(
        self,
        papers,
        include_terms: set[str],
        exclude_terms: set[str],
        human_threshold: float = 0.80,
    ) -> dict:
        # ACQUIS TD05 — Orchestrer observation, décision et file humaine.
        decisions = []
        for paper in papers:
            indicators = observe_indicators(paper, include_terms, exclude_terms)
            decision = apply_screening_policy(
                paper.id, indicators, human_threshold
            )
            decisions.append(decision)
        human_queue = [item for item in decisions if item.needs_human_review]
        return {"decisions": decisions, "human_queue": human_queue}
