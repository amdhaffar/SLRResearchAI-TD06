from pydantic import ValidationError

from .contracts import AgentAction, Observation, SearchState
from .policies import DecisionPolicy
from .runtime import execute_authorized_action, update_state
from .tools import ToolRegistry


def run_react_agent(
    objective: str,
    queries: dict[str, str],
    registry: ToolRegistry,
    decide: DecisionPolicy,
    max_steps: int = 5,
    target_papers: int = 3,
) -> SearchState:
    """Implémente la boucle ReAct contrôlée complète.

    À chaque tour : décider, valider, traiter finish ou exécuter, observer,
    mettre à jour. Les actions invalides/refusées deviennent des observations.
    La boucle doit aussi s'arrêter sur target_papers ou max_steps.
    """
    state = SearchState(objective=objective, queries=queries)

    for _ in range(max_steps):
        raw_action = decide(state, registry.names)
        try:
            action = AgentAction.model_validate(raw_action)
        except (ValidationError, ValueError) as exc:
            observation = Observation(
                status="invalid_action",
                error=str(exc),
                raw_action=raw_action if isinstance(raw_action, dict) else None,
            )
            update_state(state, None, observation, raw_action=raw_action)
            continue

        if action.action == "finish":
            state.actions_executed.append(action)
            state.step += 1
            state.finished = True
            state.stop_reason = "agent_finish"
            return state

        observation = execute_authorized_action(action, registry)
        update_state(state, action, observation)

        if len(state.papers) >= target_papers:
            state.finished = True
            state.stop_reason = "target_reached"
            return state

    state.finished = True
    state.stop_reason = "max_steps"
    return state
