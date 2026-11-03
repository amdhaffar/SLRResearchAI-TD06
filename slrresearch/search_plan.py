"""Acquis complet du TD02, conservé dans le projet. Ne pas modifier au TD03Bis."""

import re
from typing import Literal

from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field


class ResearchQuestion(BaseModel):
    """Contrat des données utilisables après le contrôle textuel."""

    identifier: str = Field(pattern=r"^RQ[0-9]+$")
    question: str = Field(min_length=15)


class WorkflowOutcome(BaseModel):
    """État observable du workflow après validation et repair éventuel."""

    status: Literal["accepted", "repaired", "human_required"]
    text: str
    questions: list[ResearchQuestion] = Field(default_factory=list)
    error: str | None = None
    repair_attempts: int = 0


RQ_PATTERN = re.compile(r"^(RQ[0-9]+)\.\s*(.{15,})$")
SEARCH_CONCEPTS = {
    "task": [
        "systematic review screening",
        "study selection",
        "title and abstract screening",
    ],
    "technology": ["large language models", "LLM", "generative AI"],
}


def extract_non_empty_lines(text: str) -> list[str]:
    """Nettoie le texte et retourne les lignes non vides."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("Texte vide")
    return lines


def parse_rq_line(line: str) -> dict:
    """Contrôle toute la ligne et extrait son identifiant et sa question."""
    match = RQ_PATTERN.fullmatch(line)
    if not match:
        raise ValueError(f"Ligne non conforme : {line}")
    return {
        "identifier": match.group(1),
        "question": match.group(2).strip(),
    }

def validate_identifiers(records: list[dict]) -> None:
    """Exige exactement RQ1, RQ2 et RQ3 dans cet ordre."""
    identifiers = [record["identifier"] for record in records]
    if identifiers != ["RQ1", "RQ2", "RQ3"]:
        raise ValueError("Identifiants attendus : RQ1, RQ2, RQ3")


def validate_no_duplicates(records: list[dict]) -> None:
    """Refuse deux formulations identiques."""
    questions = [record["question"] for record in records]
    if len(set(questions)) != len(questions):
        raise ValueError("Les trois questions doivent être différentes")


def validate_free_text(text: str) -> list[dict]:
    """Compose les contrôles textuels déterministes."""
    lines = extract_non_empty_lines(text)
    records = [parse_rq_line(line) for line in lines]
    validate_identifiers(records)
    validate_no_duplicates(records)
    return records


def build_few_shot_prompt(topic: str) -> str:
    """Construit un prompt avec deux exemples de sorties conformes."""
    clean = topic.strip()
    if not clean:
        raise ValueError("Sujet vide")
    return f"""Propose exactement trois questions de recherche.
N'ajoute ni introduction, ni conclusion, ni JSON.

Exemple 1 - Sujet : IA et éducation
RQ1. Comment l'IA influence-t-elle la personnalisation pédagogique ?
RQ2. Quels risques l'IA introduit-elle dans l'évaluation ?
RQ3. Quel contrôle humain doit-il être conservé ?

Exemple 2 - Sujet : DevOps et qualité
RQ1. Comment DevOps influence-t-il la fréquence des déploiements ?
RQ2. Quels facteurs réduisent-ils les défauts en production ?
RQ3. Quelles métriques évaluent-elles cette amélioration ?

Sujet courant : {clean}
Retourne exactement trois lignes RQ1., RQ2. et RQ3."""


def build_repair_prompt(text: str, error: str) -> str:
    """Demande au LLM une correction de forme ciblée."""
    return f"""Tu corriges la forme d'une sortie destinée à un programme.

[TÂCHE]
Corrige uniquement la forme du texte.
Préserve le sens des trois questions.

[ERREUR DÉTECTÉE]
{error}

[CONTRAT ATTENDU]
- Exactement trois lignes.
- Chaque ligne commence respectivement par RQ1., RQ2. et RQ3.
- N'ajoute ni introduction, ni explication, ni Markdown, ni JSON.

[EXEMPLE CONFORME]
RQ1. Comment l'IA influence-t-elle la personnalisation pédagogique ?
RQ2. Quels risques l'IA introduit-elle dans l'évaluation ?
RQ3. Quel contrôle humain doit-il être conservé ?

[TEXTE À CORRIGER]
{text}

[SORTIE]
"""


def repair_with_llm(model, text: str, error: str) -> str:
    """Code fourni : un seul appel conditionnel de réparation."""
    response = model.invoke(build_repair_prompt(text, error))
    return str(response.content).strip()


def validate_questions(records: list[dict]) -> list[ResearchQuestion]:
    """Applique le contrat Pydantic à chaque dictionnaire."""
    return [ResearchQuestion.model_validate(record) for record in records]
     


def build_queries(questions: list[ResearchQuestion]) -> dict[str, str]:
    """Produit des requêtes combinant synonymes OR et concepts AND."""
    if not questions:
        raise ValueError("Au moins une RQ validée est nécessaire")
    combinations = []
    for task in SEARCH_CONCEPTS["task"]:
        for technology in SEARCH_CONCEPTS["technology"]:
            expression = f'("{task}" AND "{technology}")'
            combinations.append(expression)
    base_query = " OR ".join(combinations)
    return {
        "scholar": base_query,
        "ieee": base_query,
        "acm": base_query,
    }


def build_question_pipeline():
    """Compose regex | Pydantic | requêtes avec LCEL."""
    text_control = RunnableLambda(validate_free_text)
    data_contract = RunnableLambda(validate_questions)
    query_builder = RunnableLambda(build_queries)
    return text_control | data_contract | query_builder


def run_controlled_workflow(model, text: str) -> WorkflowOutcome:
    """Valide, répare une fois, revalide ou escalade vers un humain."""
    try:
        # Code fourni - chemin nominal : aucune réparation si le texte est valide.
        records = validate_free_text(text)
        questions = validate_questions(records)
        return WorkflowOutcome(
            status="accepted",
            text=text,
            questions=questions,
            repair_attempts=0,
            error=None,
        )

    except ValueError as first_error:
        # Code fourni - une seule tentative de réparation après l'échec.
        repaired_text = repair_with_llm(
            model,
            text,
            str(first_error),
        )
        try:
            # Correction 7.1 : le texte réparé repasse par exactement les
            # mêmes contrôles déterministes et Pydantic que le texte initial.
            repaired_records = validate_free_text(repaired_text)
            repaired_questions = validate_questions(repaired_records)

            return WorkflowOutcome(
                status="repaired",
                text=repaired_text,
                questions=repaired_questions,
                repair_attempts=1,
                error=None,
            )

        except ValueError as second_error:
            # Correction 7.2 : après l'unique repair autorisé, le système
            # s'abstient et transmet la décision à un humain.
            return WorkflowOutcome(
                status="human_required",
                text=repaired_text,
                questions=[],
                repair_attempts=1,
                error=str(second_error),
            )
