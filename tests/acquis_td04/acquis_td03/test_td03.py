from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from slrresearch.contracts import AgentAction, Observation, PaperRecord, SearchState
from slrresearch.policies import LLMDecisionPolicy, ScriptedPolicy, build_decision_prompt
from slrresearch.react_agent import run_react_agent
from slrresearch.runtime import execute_authorized_action, update_state
from slrresearch.tools import SearchTool, ToolNotAllowedError, ToolRegistry
from slrresearch.workflow import run_tool_workflow


QUERY = '("systematic review" AND "large language models")'


def paper(identifier: str, doi: str = "10.1/shared") -> dict:
    return {
        "id": identifier,
        "title": f"Article {identifier}",
        "abstract": "LLM screening",
        "doi": doi,
        "url": f"https://example.org/{identifier}",
        "open_access": True,
    }


def registry_from(backends: dict[str, object]) -> ToolRegistry:
    return ToolRegistry([SearchTool(name, backend) for name, backend in backends.items()])


def test_todo_1_search_exige_des_arguments_complets():
    valid = AgentAction(action="search", tool="openalex", query=QUERY, limit=5)
    assert valid.tool == "openalex"
    with pytest.raises(ValidationError):
        AgentAction(action="search", tool="openalex", query="", limit=5)


def test_todo_1_finish_interdit_les_arguments_outil():
    assert AgentAction(action="finish").tool is None
    with pytest.raises(ValidationError):
        AgentAction(action="finish", tool="openalex")


def test_todo_2_registry_applique_la_liste_blanche():
    registry = registry_from({"openalex": lambda q, n: []})
    assert registry.authorize("openalex").name == "openalex"
    with pytest.raises(ToolNotAllowedError):
        registry.authorize("outil_interdit")


def test_todo_4_execution_distingue_success_et_empty():
    registry = registry_from(
        {
            "with_results": lambda q, n: [paper("P1")],
            "empty": lambda q, n: [],
        }
    )
    success = execute_authorized_action(
        AgentAction(action="search", tool="with_results", query=QUERY, limit=5),
        registry,
    )
    empty = execute_authorized_action(
        AgentAction(action="search", tool="empty", query=QUERY, limit=5),
        registry,
    )
    assert success.status == "success" and success.count == 1
    assert empty.status == "empty" and empty.count == 0


def test_todo_4_execution_transforme_http_429_en_observation():
    request = httpx.Request("GET", "https://example.org")
    response = httpx.Response(429, request=request)

    def limited(q, n):
        raise httpx.HTTPStatusError("limited", request=request, response=response)

    registry = registry_from({"limited": limited})
    observation = execute_authorized_action(
        AgentAction(action="search", tool="limited", query=QUERY, limit=5),
        registry,
    )
    assert observation.status == "rate_limited"
    assert observation.error == "HTTP 429"


def test_todo_5_etat_conserve_trace_erreur_et_deduplication():
    state = SearchState(objective="chercher", queries={"a": QUERY})
    action = AgentAction(action="search", tool="a", query=QUERY, limit=5)
    duplicate_a = PaperRecord(source="a", **paper("P1"))
    duplicate_b = PaperRecord(source="b", **paper("P2"))
    update_state(
        state,
        action,
        Observation(status="success", tool="a", count=2, papers=[duplicate_a, duplicate_b]),
    )
    assert state.step == 1 and len(state.trace) == 1
    assert len(state.papers) == 1


def test_todo_6_workflow_programme_choisit_tous_les_outils_dans_l_ordre():
    calls = []

    def backend(name):
        def search(q, n):
            calls.append(name)
            return []

        return search

    registry = registry_from({name: backend(name) for name in ["a", "b", "c"]})
    state = run_tool_workflow(
        "objectif",
        {"a": "qa", "b": "qb", "c": "qc"},
        registry,
    )
    assert calls == ["a", "b", "c"]
    assert state.stop_reason == "workflow_complete" and state.step == 3


def test_todo_6_workflow_continue_apres_une_source_limitee():
    request = httpx.Request("GET", "https://example.org")
    response = httpx.Response(429, request=request)

    def limited(q, n):
        raise httpx.HTTPStatusError("limited", request=request, response=response)

    registry = registry_from({"limited": limited, "working": lambda q, n: [paper("P1")]})
    state = run_tool_workflow(
        "objectif",
        {"limited": QUERY, "working": QUERY},
        registry,
    )
    assert [item.status for item in state.observations] == ["rate_limited", "success"]
    assert len(state.papers) == 1


def test_todo_7_prompt_expose_etat_contrat_et_outils_autorises():
    state = SearchState(objective="Trouver des études", queries={"openalex": QUERY})
    prompt = build_decision_prompt(state, ["openalex", "crossref"])
    assert "Trouver des études" in prompt
    assert "openalex" in prompt and "crossref" in prompt
    assert '"action":"search"' in prompt and '"action":"finish"' in prompt
    assert "n'exécutes jamais" in prompt


def test_todo_7_policy_effectue_un_appel_et_extrait_json():
    class FakeModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt):
            self.calls += 1
            return SimpleNamespace(
                content='```json\n{"action":"search","tool":"openalex","query":"q","limit":5}\n```'
            )

    model = FakeModel()
    state = SearchState(objective="objectif", queries={"openalex": "q"})
    decision = LLMDecisionPolicy(model)(state, ["openalex"])
    assert decision["tool"] == "openalex" and model.calls == 1


def test_todo_8_finish_n_execute_aucun_outil():
    calls = []
    registry = registry_from({"openalex": lambda q, n: calls.append(q) or []})
    policy = ScriptedPolicy([{"action": "finish"}])
    state = run_react_agent("objectif", {"openalex": QUERY}, registry, policy)
    assert calls == []
    assert state.finished and state.stop_reason == "agent_finish"


def test_todo_8_observation_429_permet_une_action_adaptee():
    request = httpx.Request("GET", "https://example.org")
    response = httpx.Response(429, request=request)

    def limited(q, n):
        raise httpx.HTTPStatusError("limited", request=request, response=response)

    registry = registry_from(
        {"limited": limited, "alternative": lambda q, n: [paper("P1", "10.1/p1")]}
    )
    policy = ScriptedPolicy(
        [
            {"action": "search", "tool": "limited", "query": QUERY, "limit": 5},
            {"action": "search", "tool": "alternative", "query": QUERY, "limit": 5},
            {"action": "finish"},
        ]
    )
    state = run_react_agent(
        "objectif",
        {"limited": QUERY, "alternative": QUERY},
        registry,
        policy,
        max_steps=5,
        target_papers=3,
    )
    assert [item.status for item in state.observations] == ["rate_limited", "success"]
    assert len(state.papers) == 1 and state.stop_reason == "agent_finish"


def test_todo_8_action_invalide_devient_observation_et_la_boucle_continue():
    registry = registry_from({"openalex": lambda q, n: []})
    policy = ScriptedPolicy(
        [
            {"action": "search", "tool": "openalex", "query": "", "limit": 5},
            {"action": "finish"},
        ]
    )
    state = run_react_agent("objectif", {"openalex": QUERY}, registry, policy)
    assert state.observations[0].status == "invalid_action"
    assert state.stop_reason == "agent_finish"


def test_todo_8_outil_interdit_devient_observation_sans_execution():
    registry = registry_from({"openalex": lambda q, n: []})
    policy = ScriptedPolicy(
        [
            {"action": "search", "tool": "shell", "query": QUERY, "limit": 5},
            {"action": "finish"},
        ]
    )
    state = run_react_agent("objectif", {"openalex": QUERY}, registry, policy)
    assert state.observations[0].status == "unauthorized"
    assert state.stop_reason == "agent_finish"


def test_todo_8_target_reached_arrete_avant_les_actions_restantes():
    registry = registry_from({"openalex": lambda q, n: [paper("P1", "10.1/p1")]})
    policy = ScriptedPolicy(
        [
            {"action": "search", "tool": "openalex", "query": QUERY, "limit": 5},
            {"action": "search", "tool": "openalex", "query": QUERY, "limit": 5},
        ]
    )
    state = run_react_agent(
        "objectif", {"openalex": QUERY}, registry, policy, target_papers=1
    )
    assert state.stop_reason == "target_reached"
    assert policy.calls == 1


def test_todo_8_max_steps_borne_un_agent_qui_ne_termine_pas():
    registry = registry_from({"empty": lambda q, n: []})

    def never_finish(state, allowed_tools):
        return {"action": "search", "tool": "empty", "query": QUERY, "limit": 5}

    state = run_react_agent(
        "objectif",
        {"empty": QUERY},
        registry,
        never_finish,
        max_steps=3,
        target_papers=99,
    )
    assert state.stop_reason == "max_steps" and state.step == 3

