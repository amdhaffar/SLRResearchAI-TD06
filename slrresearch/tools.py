from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from html import unescape
import json
import re

import httpx


SearchCallable = Callable[[str, int], list[dict]]


def clean_abstract(value: str | None) -> str:
    """Retire les balises JATS/HTML souvent renvoyées par Crossref."""
    text = re.sub(r"<[^>]+>", " ", value or "")
    return " ".join(unescape(text).split())


def rebuild_openalex_abstract(index: dict | None) -> str:
    """Reconstruit le résumé depuis l'index inversé OpenAlex."""
    if not index:
        return ""
    positions = [(position, word) for word, occurrences in index.items() for position in occurrences]
    return " ".join(word for _, word in sorted(positions))


@dataclass(frozen=True)
class SearchTool:
    name: str
    search: SearchCallable


class ToolNotAllowedError(ValueError):
    pass


class ToolRegistry:
    """Liste blanche des outils que le runtime autorise."""

    def __init__(self, tools: list[SearchTool]):
        self._tools = {tool.name: tool for tool in tools}

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def authorize(self, tool_name: str | None) -> SearchTool:
        """Retourne seulement un outil enregistré."""
        if tool_name is None or tool_name not in self._tools:
            raise ToolNotAllowedError(f"Outil non autorisé : {tool_name}")
        return self._tools[tool_name]


def load_catalog_backend(path: Path, source: str) -> SearchCallable:
    """Backend local reproductible utilisé par les tests et la démonstration."""

    records = json.loads(path.read_text(encoding="utf-8"))

    def search(query: str, limit: int) -> list[dict]:
        tokens = {
            token.lower().strip('()"')
            for token in query.replace("AND", " ").replace("OR", " ").split()
            if len(token.strip('()"')) > 2
        }
        matches = []
        for item in records:
            if item["source"] != source:
                continue
            haystack = f'{item["title"]} {item.get("abstract", "")}'.lower()
            if not tokens or any(token in haystack for token in tokens):
                matches.append({key: value for key, value in item.items() if key != "source"})
        return matches[:limit]

    return search


def openalex_search(query: str, limit: int) -> list[dict]:
    response = httpx.get(
        "https://api.openalex.org/works",
        params={"search": query, "per-page": limit},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    return [
        {
            "id": item["id"].rsplit("/", 1)[-1],
            "title": item.get("title") or "Sans titre",
            "abstract": rebuild_openalex_abstract(item.get("abstract_inverted_index")),
            "doi": item.get("doi") or "",
            "url": (
                (item.get("best_oa_location") or {}).get("pdf_url")
                or (item.get("primary_location") or {}).get("landing_page_url")
                or item["id"]
            ),
            "open_access": bool((item.get("open_access") or {}).get("is_oa")),
        }
        for item in data.get("results", [])
    ]


def crossref_search(query: str, limit: int) -> list[dict]:
    response = httpx.get(
        "https://api.crossref.org/works",
        params={"query.bibliographic": query, "rows": limit},
        timeout=20,
    )
    response.raise_for_status()
    items = response.json()["message"]["items"]
    return [
        {
            "id": item.get("DOI") or item.get("URL", ""),
            "title": (item.get("title") or ["Sans titre"])[0],
            "abstract": clean_abstract(item.get("abstract")),
            "doi": item.get("DOI", "") or "",
            "url": item.get("URL", "") or "",
            "open_access": bool(item.get("license")),
        }
        for item in items
    ]
