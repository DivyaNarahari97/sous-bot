"""Recipe search via Tavily for ingredient grounding.

T3 uses this to look up real recipes and extract accurate ingredient lists
when scanning or when the voice assistant needs to know what ingredients
a dish requires. This supplements vision detection with web knowledge.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient

load_dotenv()


class RecipeInfo(BaseModel):
    """A recipe found via web search."""

    title: str = ""
    url: str = ""
    ingredients: list[str] = Field(default_factory=list)
    snippet: str = ""


class RecipeSearcher:
    """Searches for recipes using Tavily to ground ingredient lists."""

    def __init__(self, api_key: str | None = None) -> None:
        self._client = TavilyClient(
            api_key=api_key or os.environ["TAVILY_API_KEY"]
        )

    def search_recipe(self, dish_name: str) -> list[RecipeInfo]:
        """Search for a recipe and return results with snippets."""
        response = self._client.search(
            query=f"{dish_name} recipe ingredients list",
            max_results=3,
            search_depth="basic",
        )
        results: list[RecipeInfo] = []
        for item in response.get("results", []):
            results.append(RecipeInfo(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", "")[:500],
            ))
        return results

    def get_ingredients_for_dish(self, dish_name: str) -> list[str]:
        """Search for a dish and extract ingredient names using LLM."""
        results = self.search_recipe(dish_name)
        if not results:
            return []

        # Combine snippets for context
        context = "\n".join(
            f"Source: {r.title}\n{r.snippet}" for r in results[:3]
        )

        # Use Nebius LLM to extract clean ingredient list from search results
        from openai import OpenAI

        client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=os.environ["NEBIUS_API_KEY"],
        )

        resp = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract a simple list of ingredient NAMES from the recipe info below. "
                        "Return ONLY a comma-separated list of ingredient names, lowercase, no quantities. "
                        "Example: pasta, eggs, parmesan, black pepper, guanciale"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Recipe: {dish_name}\n\n{context}",
                },
            ],
            max_tokens=200,
            temperature=0.1,
        )

        raw = resp.choices[0].message.content or ""
        # Parse comma-separated list
        ingredients = [
            i.strip().lower()
            for i in raw.split(",")
            if i.strip() and len(i.strip()) < 50
        ]
        return ingredients
