from __future__ import annotations

from typing import Callable

from .agentic_rag_models import (
    AdaptiveRAGState,
    AgentTraceEvent,
    AgenticCase,
    ComparisonReport,
    EvidenceFacet,
    RetrievalAction,
)
from .rag_models import Chunk
from .rag_pipeline import build_context, retrieval_metrics, verify_evidence


RetrieverTool = Callable[[list[Chunk], str, int], list[dict]]
Decider = Callable[[AdaptiveRAGState], RetrievalAction]


def evaluate_facets(
    candidates: list[dict], facets: list[EvidenceFacet]
) -> tuple[list[str], list[str]]:
    """Évalue la couverture des aspects attendus dans les passages récupérés."""
    # TODO 1 — Remplacer seulement None par l'attribut demandé dans l'énoncé.
    corpus = " ".join(item["text"] for item in candidates).casefold()
    covered = []
    missing = []
    for facet in facets:
        is_covered = any(
            keyword.casefold() in corpus for keyword in facet.keywords
        )
        destination = covered if is_covered else missing
        destination.append(None)  # TODO 1 : identifiant de la facette
    return covered, missing


def controlled_decider(state: AdaptiveRAGState) -> RetrievalAction:
    """Politique reproductible : la décision dépend de l'observation courante."""
    # TODO 2 — Compléter uniquement les quatre décisions indiquées dans l'énoncé.
    if state.iteration == 0:
        return RetrievalAction(
            action="retrieve",
            tool="lexical",
            query=state.question,
            reason="Commencer par la baseline lexicale.",
        )
    if not state.missing_facets:
        return RetrievalAction(action="finish", reason="À compléter")
    if state.iteration >= state.max_iterations:
        return RetrievalAction(action="abstain", reason="À compléter")
    missing = {facet.id for facet in state.facets} & set(state.missing_facets)
    refined_query = " ".join(
        facet.query_hint for facet in state.facets if facet.id in missing
    )
    return RetrievalAction(
        action="retrieve",
        tool=None,
        query=refined_query,
        reason="À compléter",
    )


def execute_retrieval_action(
    action: RetrievalAction,
    tools: dict[str, RetrieverTool],
    chunks: list[Chunk],
    k: int,
) -> list[dict]:
    """Contrôle l'autorisation puis exécute exactement un outil de retrieval."""
    # TODO 3 — Vérifier l'action, vérifier l'outil, puis l'appeler.
    if action.action != "retrieve" or action.tool is None:
        raise ValueError("Cette action ne demande pas de retrieval")
    if action.tool not in tools:
        raise KeyError(f"Outil non autorisé : {action.tool}")
    return []


def merge_candidates(
    previous: list[dict], new: list[dict], limit: int
) -> list[dict]:
    """Fusionne les cycles sans dupliquer un chunk."""
    # TODO 4 — Compléter les trois zones marquées.
    by_id: dict[str, dict] = {}
    for item in [*previous, *new]:
        chunk_id = item["chunk_id"]
        if ( 
            chunk_id not in by_id 
            or ______
        ):
            by_id[chunk_id] = item
    return sorted(
        by_id.values(),
        key=lambda item: ______,
        reverse=True,
    )[:______]


def update_state_after_retrieval(
    state: AdaptiveRAGState,
    action: RetrievalAction,
    results: list[dict],
    memory_limit: int,
) -> None:
    """Met à jour l'état partagé et ajoute une trace compacte."""
    # TODO 5 — Remplacer seulement les quatre valeurs incomplètes.
    state.selected_chunks = merge_candidates(
        state.selected_chunks, results, memory_limit
    )
    covered, missing = evaluate_facets(state.selected_chunks, state.facets)
    state.iteration += 1
    state.retrieval_calls += 1
    state.covered_facets = None
    state.missing_facets = None
    state.trace.append(
        AgentTraceEvent(
            iteration=state.iteration,
            action=action.action,
            tool=action.tool,
            query=action.query,
            found=len(results),
            covered_facets=[],
            missing_facets=[],
            reason=action.reason,
        )
    )


def run_adaptive_retrieval(
    question: str,
    facets: list[EvidenceFacet],
    chunks: list[Chunk],
    tools: dict[str, RetrieverTool],
    decider: Decider = controlled_decider,
    k: int = 1,
    max_iterations: int = 2,
    memory_limit: int = 4,
) -> AdaptiveRAGState:
    """Boucle bornée : décider → agir → observer → réévaluer."""
    state = AdaptiveRAGState(
        question=question,
        facets=facets,
        max_iterations=max_iterations,
        missing_facets=[facet.id for facet in facets],
    )
    # TODO 6 — Compléter uniquement le corps de la boucle avec le squelette de l'énoncé.
    while state.status == "running":
        action = decider(state)
        raise NotImplementedError("TODO 6")
    return state


def compare_baseline_and_agentic(
    case: AgenticCase,
    baseline_results: list[dict],
    agent_state: AdaptiveRAGState,
) -> ComparisonReport:
    """Compare qualité, couverture et coût avec les mêmes références."""
    relevant = set(case.relevant_chunk_ids)
    # TODO 7 — Calculer les métriques sur les deux listes d'identifiants.
    baseline_metrics = None
    agent_metrics = None
    baseline_covered, baseline_missing = evaluate_facets(
        baseline_results, case.facets
    )
    baseline = {
        **baseline_metrics,
        "covered_facets": baseline_covered,
        "missing_facets": baseline_missing,
        "retrieval_calls": 1,
    }
    agentic = {
        **agent_metrics,
        "covered_facets": agent_state.covered_facets,
        "missing_facets": agent_state.missing_facets,
        "retrieval_calls": agent_state.retrieval_calls,
        "status": agent_state.status,
    }
    return ComparisonReport(
        case_id=case.id,
        baseline=baseline,
        agentic=agentic,
        delta_recall=agent_metrics["recall_at_k"]
        - baseline_metrics["recall_at_k"],
        extra_retrieval_calls=agent_state.retrieval_calls - 1,
    )


def build_llm_decider(model) -> Decider:
    """Variante réelle : le LLM choisit une action dans le contrat autorisé."""
    structured_model = model.with_structured_output(RetrievalAction)

    def decide(state: AdaptiveRAGState) -> RetrievalAction:
        facet_details = "\n".join(
            f"- id={facet.id}; label={facet.label}; "
            f"mots-clés={facet.keywords}; requête conseillée={facet.query_hint}"
            for facet in state.facets
            if facet.id in state.missing_facets
        )
        prompt = f"""Tu pilotes un retrieval adaptatif sous contraintes.
Actions autorisées : retrieve avec lexical ou semantic, finish, abstain.
Question : {state.question}
Itération : {state.iteration}/{state.max_iterations}
Facettes couvertes : {state.covered_facets}
Facettes manquantes : {state.missing_facets}
Détails des facettes manquantes :
{facet_details or "Aucune facette manquante."}
La première recherche lexicale a déjà été exécutée par le programme.
Si une facette manque et que le budget le permet, choisis semantic et construis
une requête ciblée à partir de sa requête conseillée. Ne répète pas la question
française initiale.
RÈGLES OBLIGATOIRES :
- retrieve exige tool="lexical" ou tool="semantic" ET une query non vide ;
- finish et abstain exigent tool=null et query="" ;
- reason doit expliquer la décision ;
- n'invente aucun autre outil.
Exemple retrieve :
{{"action":"retrieve","tool":"semantic","query":"human oversight quality control reproducibility","reason":"la recherche lexicale n'a pas couvert la facette human_oversight"}}
Exemple terminal :
{{"action":"finish","tool":null,"query":"","reason":"toutes les facettes sont couvertes"}}"""
        # TODO 8 — Remplacer les pointillés par un seul appel structuré.
        return ...

    return decide


def finalize_agentic_evidence(state: AdaptiveRAGState, extractor):
    """Réutilise la génération et la vérification du TD05 après l'arrêt."""
    if state.status != "completed" or not state.selected_chunks:
        return None
    context = build_context(state.selected_chunks)
    proposal = extractor(state.question, context)
    return verify_evidence(proposal, state.selected_chunks)
