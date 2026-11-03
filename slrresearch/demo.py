from pathlib import Path

import httpx

from .tools import SearchTool, ToolRegistry, load_catalog_backend


def build_demo_registry(simulate_429: bool = True) -> ToolRegistry:
    catalog = Path(__file__).resolve().parent.parent / "data" / "papers.json"

    if simulate_429:
        request = httpx.Request("GET", "https://api.openalex.org/works")
        response = httpx.Response(429, request=request)

        def openalex_backend(query: str, limit: int) -> list[dict]:
            raise httpx.HTTPStatusError(
                "Rate limit pédagogique",
                request=request,
                response=response,
            )
    else:
        openalex_backend = load_catalog_backend(catalog, "openalex")

    return ToolRegistry(
        [
            SearchTool("openalex", openalex_backend),
            SearchTool("crossref", load_catalog_backend(catalog, "crossref")),
            SearchTool(
                "semantic_scholar",
                load_catalog_backend(catalog, "semantic_scholar"),
            ),
        ]
    )


def build_demo_policy(queries: dict[str, str]):
    """Politique déterministe, mais réellement conditionnée par l'observation.

    - Premier tour : essayer OpenAlex.
    - Si OpenAlex échoue ou ne retourne rien : essayer Crossref.
    - Si OpenAlex réussit : terminer immédiatement.
    - Après Crossref : terminer.
    """

    def decide(state, allowed_tools: list[str]) -> dict:
        called_tools = [
            action.tool for action in state.actions_executed if action.tool is not None
        ]
        if "openalex" not in called_tools:
            return {
                "action": "search",
                "tool": "openalex",
                "query": queries["openalex"],
                "limit": 5,
            }

        last_status = state.observations[-1].status if state.observations else None
        recoverable = {
            "empty",
            "rate_limited",
            "http_error",
            "network_error",
            "unauthorized",
        }
        if last_status in recoverable and "crossref" not in called_tools:
            return {
                "action": "search",
                "tool": "crossref",
                "query": queries["crossref"],
                "limit": 5,
            }

        return {"action": "finish"}

    return decide
