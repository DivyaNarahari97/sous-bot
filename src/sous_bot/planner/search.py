from __future__ import annotations

from dataclasses import dataclass
import os

from tavily import TavilyClient


@dataclass(frozen=True)
class RecipeSource:
    title: str
    url: str
    snippet: str


def search_recipes(query: str, max_results: int = 3) -> list[RecipeSource]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=f"{query} recipe",
        max_results=max_results,
        include_answer=False,
        include_raw_content=False,
    )

    results = []
    for item in response.get("results", []):
        title = item.get("title") or query
        url = item.get("url") or ""
        snippet = item.get("content") or ""
        results.append(RecipeSource(title=title, url=url, snippet=snippet))

    return results
