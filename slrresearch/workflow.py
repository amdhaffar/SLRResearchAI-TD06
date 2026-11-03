from .contracts import AgentAction, SearchState
from .runtime import execute_authorized_action, update_state
from .tools import ToolRegistry


def run_tool_workflow(
    objective: str,
    queries: dict[str, str],
    registry: ToolRegistry,
    limit: int = 5,
) -> SearchState:
    """Baseline : le programme choisit tous les outils dans l'ordre."""
    state = SearchState(objective=objective, queries=queries)
    for tool_name, query in queries.items():
        action = AgentAction(
            action="search",
            tool=tool_name,
            query=query,
            limit=limit,
        )
        observation = execute_authorized_action(action, registry)
        update_state(state, action, observation)
    state.finished = True
    state.stop_reason = "workflow_complete"
    return state
