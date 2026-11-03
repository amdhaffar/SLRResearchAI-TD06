from time import perf_counter

import httpx

from .contracts import AgentAction, Observation, PaperRecord, SearchState, TraceEvent
from .tools import ToolNotAllowedError, ToolRegistry


def execute_authorized_action(
    action: AgentAction,
    registry: ToolRegistry,
) -> Observation:
    """Autorise, exécute et convertit toute issue en Observation."""
    started = perf_counter()
    try:
        tool = registry.authorize(action.tool)
    except ToolNotAllowedError as exc:
        return Observation(
            tool=action.tool,
            status="unauthorized",
            latency_seconds=perf_counter() - started,
            error=str(exc),
        )

    try:
        raw_papers = tool.search(action.query or "", action.limit or 1)
        papers = [
            PaperRecord(source=tool.name, **raw_paper)
            for raw_paper in raw_papers
        ]
        return Observation(
            tool=tool.name,
            status="success" if papers else "empty",
            count=len(papers),
            latency_seconds=perf_counter() - started,
            papers=papers,
        )
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        return Observation(
            tool=tool.name,
            status="rate_limited" if status_code == 429 else "http_error",
            latency_seconds=perf_counter() - started,
            error=f"HTTP {status_code}",
        )
    except httpx.HTTPError as exc:
        return Observation(
            tool=tool.name,
            status="network_error",
            latency_seconds=perf_counter() - started,
            error=str(exc),
        )


def paper_key(paper: PaperRecord) -> str:
    return (paper.doi or paper.title).lower().strip()


def update_state(
    state: SearchState,
    action: AgentAction | None,
    observation: Observation,
    raw_action: dict | None = None,
) -> None:
    """Met à jour trace, erreurs et articles sans doublons."""
    state.step += 1
    if action is not None:
        state.actions_executed.append(action)
    state.observations.append(observation)
    action_dump = raw_action or (action.model_dump() if action else {})
    state.trace.append(
        TraceEvent(step=state.step, action=action_dump, observation=observation)
    )
    if observation.error:
        state.errors.append(observation.error)

    unique = {paper_key(paper): paper for paper in state.papers}
    for paper in observation.papers:
        unique.setdefault(paper_key(paper), paper)
    state.papers = list(unique.values())
