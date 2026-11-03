import pytest

import slrresearch.agentic_rag as agentic
from slrresearch.agentic_rag import (
    build_llm_decider,
    compare_baseline_and_agentic,
    controlled_decider,
    evaluate_facets,
    execute_retrieval_action,
    merge_candidates,
    run_adaptive_retrieval,
    update_state_after_retrieval,
)
from slrresearch.agentic_rag_models import (
    AdaptiveRAGState,
    AgenticCase,
    EvidenceFacet,
    RetrievalAction,
)
from slrresearch.rag_models import Chunk


FACETS = [
    EvidenceFacet(id="benefit", label="Avantage", keywords=["robustness"], query_hint="robustness"),
    EvidenceFacet(id="limit", label="Limite", keywords=["cost"], query_hint="cost latency"),
]


def candidate(chunk_id, text, score=0.5):
    return {
        "paper_id": "P1",
        "chunk_id": chunk_id,
        "page": 1,
        "text": text,
        "retrieval_score": score,
    }


def test_todo_1_evalue_la_couverture_des_facettes():
    covered, missing = evaluate_facets(
        [candidate("c1", "The method improves robustness.")], FACETS
    )
    assert covered == ["benefit"] and missing == ["limit"]


def test_todo_2_decide_selon_observation_et_budget():
    state = AdaptiveRAGState(question="q", facets=FACETS, missing_facets=["benefit", "limit"])
    assert controlled_decider(state).tool == "lexical"
    state.iteration = 1
    state.missing_facets = ["limit"]
    action = controlled_decider(state)
    assert action.tool == "semantic" and "cost" in action.query
    state.missing_facets = []
    assert controlled_decider(state).action == "finish"


def test_todo_3_execute_uniquement_un_outil_autorise():
    calls = []
    tool = lambda chunks, query, k: calls.append((query, k)) or [candidate("c1", "x")]
    action = RetrievalAction(action="retrieve", tool="lexical", query="q", reason="test")
    assert execute_retrieval_action(action, {"lexical": tool}, [], 2)[0]["chunk_id"] == "c1"
    assert calls == [("q", 2)]
    forbidden = RetrievalAction(action="retrieve", tool="semantic", query="q", reason="test")
    with pytest.raises(KeyError):
        execute_retrieval_action(forbidden, {"lexical": tool}, [], 1)


def test_todo_4_fusionne_dedoublonne_et_limite():
    merged = merge_candidates(
        [candidate("c1", "ancien", 0.2), candidate("c2", "deux", 0.6)],
        [candidate("c1", "nouveau", 0.9), candidate("c3", "trois", 0.1)],
        limit=2,
    )
    assert [item["chunk_id"] for item in merged] == ["c1", "c2"]
    assert merged[0]["text"] == "nouveau"


def test_todo_5_met_a_jour_et_trace_l_etat(monkeypatch):
    monkeypatch.setattr(agentic, "merge_candidates", lambda old, new, limit: new)
    monkeypatch.setattr(agentic, "evaluate_facets", lambda candidates, facets: (["benefit"], ["limit"]))
    state = AdaptiveRAGState(question="q", facets=FACETS)
    action = RetrievalAction(action="retrieve", tool="lexical", query="q", reason="initial")
    update_state_after_retrieval(state, action, [candidate("c1", "x")], 4)
    assert state.iteration == 1 and state.retrieval_calls == 1
    assert state.covered_facets == ["benefit"] and state.missing_facets == ["limit"]
    assert state.trace[0].tool == "lexical"


def test_todo_6_boucle_jusqu_a_finish(monkeypatch):
    actions = iter([
        RetrievalAction(action="retrieve", tool="lexical", query="q", reason="1"),
        RetrievalAction(action="finish", reason="preuves suffisantes"),
    ])
    monkeypatch.setattr(agentic, "execute_retrieval_action", lambda *args: [candidate("c1", "robustness cost")])

    def fake_update(state, action, results, limit):
        state.iteration += 1
        state.retrieval_calls += 1
        state.selected_chunks = results
        state.covered_facets = ["benefit", "limit"]
        state.missing_facets = []

    monkeypatch.setattr(agentic, "update_state_after_retrieval", fake_update)
    state = run_adaptive_retrieval(
        "q", FACETS, [], {"lexical": lambda *args: []},
        decider=lambda state: next(actions), max_iterations=2,
    )
    assert state.status == "completed" and state.retrieval_calls == 1


def test_todo_7_compare_qualite_et_cout(monkeypatch):
    monkeypatch.setattr(agentic, "evaluate_facets", lambda candidates, facets: ([], ["benefit", "limit"]))
    case = AgenticCase(
        id="C", label="cas", question="q", facets=FACETS,
        relevant_chunk_ids=["c2"], baseline_k=1,
    )
    state = AdaptiveRAGState(
        question="q", facets=FACETS, selected_chunks=[candidate("c2", "preuve")],
        retrieval_calls=2, covered_facets=["benefit", "limit"], status="completed",
    )
    report = compare_baseline_and_agentic(case, [candidate("c1", "bruit")], state)
    assert report.delta_recall == 1.0 and report.extra_retrieval_calls == 1


def test_todo_8_decideur_llm_utilise_la_sortie_structuree():
    class Structured:
        def invoke(self, prompt):
            assert "Facettes manquantes" in prompt
            return RetrievalAction(action="finish", reason="suffisant")

    class Model:
        def with_structured_output(self, schema):
            assert schema is RetrievalAction
            return Structured()

    state = AdaptiveRAGState(question="q", facets=FACETS)
    assert build_llm_decider(Model())(state).action == "finish"
