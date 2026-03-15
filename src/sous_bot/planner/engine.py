from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
import re
from typing import Any

import requests
import yaml
from openai import OpenAI
from dotenv import load_dotenv

from sous_bot.api.schemas import (
    ChatMessage,
    CookingStep,
    Ingredient,
    PlanRequest,
    PlanResponse,
    RobotAction,
    ShoppingListByRecipe,
    ShoppingItem,
)
from sous_bot.planner.prompts import CHAT_SYSTEM_PROMPT, MEAL_SUGGESTION_PROMPT, RECIPE_STEPS_PROMPT
from sous_bot.planner.prompts import WEEKLY_MULTI_SHOPPING_PROMPT, WEEKLY_SHOPPING_PROMPT
from sous_bot.planner.search import RecipeSource, search_recipes


@dataclass(frozen=True)
class PlannerConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    minimax_base_url: str
    minimax_api_key: str
    nebius_base_url: str
    nebius_api_key: str


def _resolve_env(value: str) -> str:
    if value.startswith("${") and value.endswith("}"):
        env_name = value[2:-1]
        return os.getenv(env_name, "")
    return value


@lru_cache(maxsize=1)
def load_settings() -> dict[str, Any]:
    load_dotenv()
    root = Path(__file__).resolve().parents[3]
    settings_path = root / "config" / "settings.yaml"
    with settings_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def load_planner_config() -> PlannerConfig:
    settings = load_settings()
    planner = settings.get("planner", {})
    api_key_raw = planner.get("minimax_api_key", "")
    api_key = _resolve_env(api_key_raw) if api_key_raw else os.getenv("MINIMAX_API_KEY", "")
    nebius_key_raw = planner.get("nebius_api_key", "")
    nebius_key = _resolve_env(nebius_key_raw) if nebius_key_raw else os.getenv("NEBIUS_API_KEY", "")

    return PlannerConfig(
        provider=planner.get("provider", "minimax"),
        model=planner.get("model", "MiniMax-M2.5"),
        temperature=float(planner.get("temperature", 0.3)),
        max_tokens=int(planner.get("max_tokens", 2048)),
        minimax_base_url=planner.get("minimax_base_url", "https://api.minimax.io/v1"),
        minimax_api_key=api_key,
        nebius_base_url=planner.get("nebius_base_url", "https://api.tokenfactory.nebius.com/v1/"),
        nebius_api_key=nebius_key,
    )


class PlannerEngine:
    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or load_planner_config()
        self._nebius_client: OpenAI | None = None

    def chat(self, history: list[ChatMessage], message: str, inventory: list[str]) -> str:
        system_message = ChatMessage(role="system", content=CHAT_SYSTEM_PROMPT)
        inventory_line = "None" if not inventory else ", ".join(sorted(set(inventory)))
        user_message = ChatMessage(
            role="user",
            content=f"Inventory: {inventory_line}\nUser message: {message}",
        )
        messages = [system_message, *history, user_message]
        return self._chat(messages)

    def chat_with_plan(
        self, history: list[ChatMessage], message: str, inventory: list[str]
    ) -> tuple[str, PlanResponse | None, str | None]:
        weekly_items = self._parse_weekly_requests(message)
        if weekly_items:
            if len(weekly_items) == 1:
                weekly = weekly_items[0]
                plan = self.plan_weekly(
                    recipe=weekly["recipe"],
                    days=weekly["days"],
                    servings_per_day=weekly["servings_per_day"],
                    inventory=inventory,
                )
                reply = (
                    f"Great. I planned {weekly['recipe']} for {weekly['days']} days "
                    f"at {weekly['servings_per_day']} servings per day. "
                    "Here is your weekly shopping list."
                )
                return reply, plan, weekly["recipe"]

            plan = self.plan_weekly_multi(weekly_items, inventory=inventory)
            recipe_names = ", ".join(item["recipe"] for item in weekly_items)
            reply = (
                "Great. I planned your weekly meals and combined the shopping list "
                f"for: {recipe_names}."
            )
            return reply, plan, recipe_names

        weekly = self._parse_weekly_request(message)
        if weekly:
            plan = self.plan_weekly(
                recipe=weekly["recipe"],
                days=weekly["days"],
                servings_per_day=weekly["servings_per_day"],
                inventory=inventory,
            )
            reply = (
                f"Great. I planned {weekly['recipe']} for {weekly['days']} days "
                f"at {weekly['servings_per_day']} servings per day. "
                "Here is your weekly shopping list."
            )
            return reply, plan, weekly["recipe"]

        reply = self.chat(history, message, inventory)
        return reply, None, None

    def plan(self, request: PlanRequest) -> PlanResponse:
        inventory = sorted({*request.available_ingredients, *[i.name for i in request.detected_ingredients]})
        recipe = request.recipe.strip()

        if not recipe:
            suggestion = self._chat(
                [
                    ChatMessage(role="system", content=CHAT_SYSTEM_PROMPT),
                    ChatMessage(
                        role="user",
                        content=MEAL_SUGGESTION_PROMPT.format(
                            inventory=", ".join(inventory) or "None",
                            request="Suggest a meal",
                        ),
                    ),
                ]
            )
            recipe = suggestion.split("\n", 1)[0].lstrip("- ").split(":", 1)[0].strip() or "Pantry meal"

        sources = search_recipes(recipe)
        sources_block = _format_sources(sources)

        prompt = RECIPE_STEPS_PROMPT.format(
            recipe=recipe,
            servings=request.servings,
            inventory=", ".join(inventory) or "None",
            sources=sources_block,
        )

        response_text = self._chat(
            [
                ChatMessage(role="system", content=CHAT_SYSTEM_PROMPT),
                ChatMessage(role="user", content=prompt),
            ]
        )

        plan_payload = _extract_json(response_text)
        steps = [CookingStep(**step) for step in plan_payload.get("steps", [])]
        missing = [Ingredient(**item) for item in plan_payload.get("missing_ingredients", [])]
        estimated_time = plan_payload.get("estimated_time") or "unknown"

        shopping_list = [ShoppingItem(**item) for item in plan_payload.get("shopping_list", [])]
        if not shopping_list:
            shopping_list = [
                ShoppingItem(name=item.name, quantity=item.quantity or "1", aisle=None) for item in missing
            ]

        robot_actions = [RobotAction(**item) for item in plan_payload.get("robot_actions", [])]

        return PlanResponse(
            steps=steps,
            missing_ingredients=missing,
            estimated_time=estimated_time,
            shopping_list=shopping_list,
            shopping_list_by_recipe=[ShoppingListByRecipe(recipe=recipe or "recipe", items=shopping_list)],
            robot_actions=robot_actions,
        )

    def plan_weekly(
        self,
        recipe: str,
        days: int,
        servings_per_day: int,
        inventory: list[str],
    ) -> PlanResponse:
        sources = search_recipes(recipe)
        sources_block = _format_sources(sources)

        prompt = WEEKLY_SHOPPING_PROMPT.format(
            recipe=recipe,
            days=days,
            servings_per_day=servings_per_day,
            inventory=", ".join(sorted(set(inventory))) or "None",
            sources=sources_block,
        )

        response_text = self._chat(
            [
                ChatMessage(role="system", content=CHAT_SYSTEM_PROMPT),
                ChatMessage(role="user", content=prompt),
            ]
        )

        plan_payload = _extract_json(response_text)
        missing = [Ingredient(**item) for item in plan_payload.get("missing_ingredients", [])]
        shopping_list = [ShoppingItem(**item) for item in plan_payload.get("shopping_list", [])]
        if not shopping_list:
            shopping_list = [
                ShoppingItem(name=item.name, quantity=item.quantity or "1", aisle=None) for item in missing
            ]

        return PlanResponse(
            steps=[],
            missing_ingredients=missing,
            estimated_time=f"{days} days",
            shopping_list=shopping_list,
            shopping_list_by_recipe=[ShoppingListByRecipe(recipe=recipe, items=shopping_list)],
            robot_actions=[],
        )

    def plan_weekly_multi(
        self, weekly_items: list[dict[str, Any]], inventory: list[str]
    ) -> PlanResponse:
        recipe_names = ", ".join(item["recipe"] for item in weekly_items)
        sources = search_recipes(recipe_names)
        sources_block = _format_sources(sources)

        lines = []
        for item in weekly_items:
            lines.append(
                f"- Recipe: {item['recipe']} | Days: {item['days']} | Servings per day: {item['servings_per_day']}"
            )

        prompt = WEEKLY_MULTI_SHOPPING_PROMPT.format(
            plan_items="\n".join(lines),
            inventory=", ".join(sorted(set(inventory))) or "None",
            sources=sources_block,
        )

        response_text = self._chat(
            [
                ChatMessage(role="system", content=CHAT_SYSTEM_PROMPT),
                ChatMessage(role="user", content=prompt),
            ]
        )

        plan_payload = _extract_json(response_text)
        recipes_payload = plan_payload.get("recipes", [])
        shopping_by_recipe: list[ShoppingListByRecipe] = []
        for entry in recipes_payload:
            recipe_name = entry.get("recipe") or "unknown"
            items = [ShoppingItem(**item) for item in entry.get("shopping_list", [])]
            if not items:
                missing = [Ingredient(**item) for item in entry.get("missing_ingredients", [])]
                items = [ShoppingItem(name=item.name, quantity=item.quantity or "1", aisle=None) for item in missing]
            shopping_by_recipe.append(ShoppingListByRecipe(recipe=recipe_name, items=items))

        return PlanResponse(
            steps=[],
            missing_ingredients=[],
            estimated_time="weekly plan",
            shopping_list=None,
            shopping_list_by_recipe=shopping_by_recipe,
            robot_actions=[],
        )

    def _chat(self, messages: list[ChatMessage]) -> str:
        provider = (self.config.provider or "minimax").lower()
        if provider == "minimax":
            return self._chat_minimax(messages)
        if provider == "nebius":
            return self._chat_nebius(messages)
        raise RuntimeError(f"Unsupported planner provider: {self.config.provider}")

    def _chat_minimax(self, messages: list[ChatMessage]) -> str:
        if not self.config.minimax_api_key:
            raise RuntimeError("MINIMAX_API_KEY is not set")

        payload = {
            "model": self.config.model,
            "messages": [message.model_dump() for message in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        endpoint = _join_url(self.config.minimax_base_url, "/chat/completions")
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {self.config.minimax_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        if not response.ok:
            raise RuntimeError(f"MiniMax API error {response.status_code}: {response.text}")

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return message.get("content") or ""

    def _chat_nebius(self, messages: list[ChatMessage]) -> str:
        if not self.config.nebius_api_key:
            raise RuntimeError("NEBIUS_API_KEY is not set")

        if self._nebius_client is None:
            self._nebius_client = OpenAI(
                api_key=self.config.nebius_api_key,
                base_url=self.config.nebius_base_url,
            )

        nebius_messages = [
            {"role": message.role, "content": [{"type": "text", "text": message.content}]}
            for message in messages
        ]
        response = self._nebius_client.chat.completions.create(
            model=self.config.model,
            messages=nebius_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content or ""

    def _parse_weekly_request(self, message: str) -> dict[str, Any] | None:
        if not _parse_is_weekly(message):
            return None
        recipe = _extract_recipe_name(message)
        return {
            "recipe": recipe,
            "days": _parse_days(message),
            "servings_per_day": _parse_servings_per_day(message),
        }

    def _parse_weekly_requests(self, message: str) -> list[dict[str, Any]]:
        lines = [line.strip() for line in re.split(r"[\n;]+", message) if line.strip()]
        if len(lines) <= 1:
            return []
        weekly_items = []
        for line in lines:
            if not _parse_is_weekly(line):
                continue
            recipe = _extract_recipe_name(line)
            weekly_items.append(
                {
                    "recipe": recipe,
                    "days": _parse_days(line),
                    "servings_per_day": _parse_servings_per_day(line),
                }
            )
        return weekly_items


def _extract_json(text: str) -> dict[str, Any]:
    if not text:
        return {}

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}

    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return {}


def _format_sources(sources: list[RecipeSource]) -> str:
    if not sources:
        return "- none"
    lines = [f"- {source.title}: {source.url}\n  {source.snippet}" for source in sources]
    return "\n".join(lines)


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _extract_int(match: re.Match[str] | None, default: int) -> int:
    if not match:
        return default
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return default


def _clean_recipe_text(text: str) -> str:
    cleaned = re.sub(r"\bfor\b.*$", "", text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\bservings?\b.*$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\bfor\s+a\s+week\b", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned.strip(" .")


def _extract_recipe_name(message: str) -> str:
    lowered = message.lower().strip()
    lowered = re.sub(r"^(i\s+want|i\s+need|make|cook)\s+", "", lowered)
    recipe = _clean_recipe_text(lowered)
    return recipe or message.strip()


def _parse_days(message: str) -> int:
    day_match = re.search(r"(\d+)\s*day", message, flags=re.IGNORECASE)
    if day_match:
        return _extract_int(day_match, 7)
    if re.search(r"\bweek\b", message, flags=re.IGNORECASE):
        return 7
    return 7


def _parse_servings_per_day(message: str) -> int:
    servings_match = re.search(r"(\d+)\s*(servings?|people|persons?)", message, flags=re.IGNORECASE)
    return _extract_int(servings_match, 2)


def _parse_is_weekly(message: str) -> bool:
    return bool(
        re.search(r"\bday(s)?\b", message, flags=re.IGNORECASE)
        or re.search(r"\bweek\b", message, flags=re.IGNORECASE)
    )
