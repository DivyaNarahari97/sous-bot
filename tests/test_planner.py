"""Tests for T1 Planner Engine and API.

All LLM calls are mocked — no real API calls.
Tests cover: schemas, engine parsing, chat/plan endpoints, session management.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sous_bot.api.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    CookingStep,
    DetectedIngredient,
    Ingredient,
    PlanRequest,
    PlanResponse,
    RobotAction,
    ShoppingItem,
    ShoppingListByRecipe,
    ShoppingListResponse,
)
from sous_bot.planner.engine import (
    PlannerConfig,
    PlannerEngine,
    _extract_json,
    _extract_recipe_name,
    _parse_days,
    _parse_is_weekly,
    _parse_servings_per_day,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _mock_config() -> PlannerConfig:
    return PlannerConfig(
        provider="nebius",
        model="test-model",
        temperature=0.3,
        max_tokens=2048,
        minimax_base_url="",
        minimax_api_key="",
        nebius_base_url="https://api.tokenfactory.nebius.com/v1/",
        nebius_api_key="test-key",
    )


def _mock_nebius_response(content: str) -> MagicMock:
    """Create a mock OpenAI chat completion response."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


SAMPLE_PLAN_JSON = json.dumps({
    "steps": [
        {"step_number": 1, "description": "Boil pasta", "duration_minutes": 10, "tools": ["pot"]},
        {"step_number": 2, "description": "Fry guanciale", "duration_minutes": 5, "tools": ["pan"]},
        {"step_number": 3, "description": "Mix eggs and cheese", "duration_minutes": 2, "tools": ["bowl"]},
    ],
    "missing_ingredients": [
        {"name": "guanciale", "quantity": "150g"},
        {"name": "black pepper", "quantity": "1 tsp"},
    ],
    "estimated_time": "20 minutes",
    "shopping_list": [
        {"name": "guanciale", "quantity": "150g", "aisle": "deli"},
        {"name": "black pepper", "quantity": "1 tsp", "aisle": "spices"},
    ],
    "robot_actions": [
        {"action_type": "navigate", "parameters": {"aisle": "deli"}},
        {"action_type": "grasp", "parameters": {"item": "guanciale"}},
    ],
})

SAMPLE_WEEKLY_JSON = json.dumps({
    "missing_ingredients": [
        {"name": "guanciale", "quantity": "1050g"},
    ],
    "shopping_list": [
        {"name": "guanciale", "quantity": "1050g", "aisle": "deli"},
        {"name": "black pepper", "quantity": "7 tsp", "aisle": "spices"},
    ],
})

SAMPLE_MULTI_WEEKLY_JSON = json.dumps({
    "recipes": [
        {
            "recipe": "pasta carbonara",
            "missing_ingredients": [{"name": "guanciale", "quantity": "450g"}],
            "shopping_list": [
                {"name": "guanciale", "quantity": "450g", "aisle": "deli"},
            ],
        },
        {
            "recipe": "chicken stir fry",
            "missing_ingredients": [{"name": "chicken", "quantity": "600g"}],
            "shopping_list": [
                {"name": "chicken", "quantity": "600g", "aisle": "meat"},
                {"name": "ginger", "quantity": "30g", "aisle": "produce"},
            ],
        },
    ]
})


# ── Schema tests ─────────────────────────────────────────────────────


class TestSchemas:
    """Test T1 Pydantic schema validation and serialization."""

    def test_detected_ingredient_from_t3(self) -> None:
        """T3 vision DetectedIngredient feeds into T1."""
        item = DetectedIngredient(name="pasta", confidence=0.95)
        assert item.name == "pasta"
        assert item.confidence == 0.95
        assert item.quantity is None

    def test_shopping_item(self) -> None:
        item = ShoppingItem(name="milk", quantity="1 gallon", aisle="dairy")
        assert item.aisle == "dairy"
        d = item.model_dump()
        assert d["name"] == "milk"

    def test_plan_response_full(self) -> None:
        plan = PlanResponse(
            steps=[CookingStep(step_number=1, description="Boil water", duration_minutes=5)],
            missing_ingredients=[Ingredient(name="salt", quantity="1 tsp")],
            estimated_time="10 minutes",
            shopping_list=[ShoppingItem(name="salt", quantity="1 tsp", aisle="spices")],
            shopping_list_by_recipe=[
                ShoppingListByRecipe(
                    recipe="pasta",
                    items=[ShoppingItem(name="salt", quantity="1 tsp", aisle="spices")],
                )
            ],
            robot_actions=[RobotAction(action_type="navigate", parameters={"aisle": "spices"})],
        )
        assert len(plan.steps) == 1
        assert plan.shopping_list_by_recipe[0].recipe == "pasta"

    def test_chat_request_with_detected_ingredients(self) -> None:
        req = ChatRequest(
            message="plan a meal",
            available_ingredients=["pasta", "eggs"],
            detected_ingredients=[
                DetectedIngredient(name="parmesan", confidence=0.9),
            ],
        )
        assert len(req.detected_ingredients) == 1
        assert req.detected_ingredients[0].name == "parmesan"

    def test_shopping_list_response(self) -> None:
        resp = ShoppingListResponse(
            recipes=[
                ShoppingListByRecipe(
                    recipe="carbonara",
                    items=[ShoppingItem(name="guanciale", quantity="150g")],
                )
            ],
            recipe_names=["carbonara"],
        )
        assert resp.recipe_names == ["carbonara"]


# ── Engine helper tests ──────────────────────────────────────────────


class TestEngineHelpers:
    """Test parsing helpers in planner.engine."""

    def test_extract_json_valid(self) -> None:
        text = 'Here is the plan: {"steps": [], "missing_ingredients": []}'
        result = _extract_json(text)
        assert "steps" in result

    def test_extract_json_empty(self) -> None:
        assert _extract_json("") == {}
        assert _extract_json("no json here") == {}

    def test_extract_json_markdown_wrapped(self) -> None:
        text = '```json\n{"steps": [{"step_number": 1}]}\n```'
        result = _extract_json(text)
        assert result["steps"][0]["step_number"] == 1

    def test_parse_days(self) -> None:
        assert _parse_days("pasta for 5 days") == 5
        assert _parse_days("chicken for a week") == 7
        assert _parse_days("salad for 3 days with 2 servings") == 3

    def test_parse_servings_per_day(self) -> None:
        assert _parse_servings_per_day("pasta for 7 days 3 servings") == 3
        assert _parse_servings_per_day("chicken for 5 days 4 people") == 4
        assert _parse_servings_per_day("pasta for 7 days") == 2  # default

    def test_parse_is_weekly(self) -> None:
        assert _parse_is_weekly("pasta for 7 days") is True
        assert _parse_is_weekly("chicken for a week") is True
        assert _parse_is_weekly("make pasta carbonara") is False

    def test_extract_recipe_name(self) -> None:
        assert "carbonara" in _extract_recipe_name("make pasta carbonara for 7 days")
        assert "stir fry" in _extract_recipe_name("cook chicken stir fry for 3 days")


# ── PlannerEngine tests (mocked LLM) ────────────────────────────────


class TestPlannerEngine:
    """Test PlannerEngine with mocked Nebius API."""

    def _make_engine(self, mock_openai_cls: MagicMock) -> PlannerEngine:
        engine = PlannerEngine(config=_mock_config())
        # Pre-set the client so it doesn't create a real one
        engine._nebius_client = mock_openai_cls
        return engine

    def test_chat(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_nebius_response(
            "I suggest making pasta carbonara with your ingredients!"
        )
        engine = self._make_engine(mock_client)

        reply = engine.chat([], "What can I make?", ["pasta", "eggs", "parmesan"])
        assert "carbonara" in reply.lower() or "pasta" in reply.lower()
        mock_client.chat.completions.create.assert_called_once()

    def test_chat_with_plan_no_weekly(self) -> None:
        """Non-weekly message returns chat reply, no plan."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_nebius_response(
            "You could make a nice omelette!"
        )
        engine = self._make_engine(mock_client)

        reply, plan, recipe = engine.chat_with_plan(
            [], "What should I cook tonight?", ["eggs", "cheese"]
        )
        assert "omelette" in reply.lower()
        assert plan is None
        assert recipe is None

    @patch("sous_bot.planner.engine.search_recipes", return_value=[])
    def test_chat_with_plan_weekly_single(self, mock_search: MagicMock) -> None:
        """Weekly request triggers plan_weekly and returns a plan."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_nebius_response(
            SAMPLE_WEEKLY_JSON
        )
        engine = self._make_engine(mock_client)

        reply, plan, recipe = engine.chat_with_plan(
            [], "Make pasta carbonara for 7 days with 2 servings", ["pasta", "eggs"]
        )
        assert plan is not None
        assert plan.shopping_list_by_recipe is not None
        assert len(plan.shopping_list_by_recipe) >= 1
        assert "7 days" in reply

    @patch("sous_bot.planner.engine.search_recipes", return_value=[])
    def test_chat_with_plan_weekly_multi(self, mock_search: MagicMock) -> None:
        """Multi-recipe weekly request parses correctly."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_nebius_response(
            SAMPLE_MULTI_WEEKLY_JSON
        )
        engine = self._make_engine(mock_client)

        reply, plan, recipe = engine.chat_with_plan(
            [],
            "pasta carbonara for 3 days; chicken stir fry for 4 days",
            ["pasta", "eggs"],
        )
        assert plan is not None
        assert plan.shopping_list_by_recipe is not None
        assert len(plan.shopping_list_by_recipe) == 2
        assert "carbonara" in recipe.lower() or "stir fry" in recipe.lower()

    @patch("sous_bot.planner.engine.search_recipes", return_value=[])
    def test_plan_single_recipe(self, mock_search: MagicMock) -> None:
        """plan() generates a full PlanResponse for a single recipe."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_nebius_response(
            SAMPLE_PLAN_JSON
        )
        engine = self._make_engine(mock_client)

        request = PlanRequest(
            recipe="pasta carbonara",
            servings=2,
            available_ingredients=["pasta", "eggs", "parmesan"],
        )
        plan = engine.plan(request)

        assert len(plan.steps) == 3
        assert plan.steps[0].description == "Boil pasta"
        assert len(plan.missing_ingredients) == 2
        assert plan.estimated_time == "20 minutes"
        assert plan.shopping_list is not None
        assert len(plan.shopping_list) == 2
        assert plan.shopping_list[0].aisle == "deli"
        assert plan.robot_actions is not None
        assert len(plan.robot_actions) == 2

    @patch("sous_bot.planner.engine.search_recipes", return_value=[])
    def test_plan_weekly(self, mock_search: MagicMock) -> None:
        """plan_weekly returns scaled shopping list."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_nebius_response(
            SAMPLE_WEEKLY_JSON
        )
        engine = self._make_engine(mock_client)

        plan = engine.plan_weekly(
            recipe="pasta carbonara",
            days=7,
            servings_per_day=2,
            inventory=["pasta", "eggs"],
        )
        assert plan.estimated_time == "7 days"
        assert plan.shopping_list is not None
        assert len(plan.shopping_list) >= 1
        assert plan.shopping_list_by_recipe is not None

    def test_chat_passes_inventory_context(self) -> None:
        """Verify inventory is included in the LLM prompt."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_nebius_response("OK")
        engine = self._make_engine(mock_client)

        engine.chat([], "suggest a meal", ["pasta", "eggs", "tomatoes"])

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages", [])
        # The user message should contain inventory info
        user_msg = messages[-1]
        content = user_msg["content"] if isinstance(user_msg, dict) else user_msg.content
        content_text = content if isinstance(content, str) else content[0]["text"]
        assert "pasta" in content_text.lower()


# ── API endpoint tests (mocked engine) ──────────────────────────────


class TestAPIEndpoints:
    """Test T1 FastAPI endpoints with mocked planner engine."""

    @pytest.fixture(autouse=True)
    def _reset_store(self) -> None:
        """Reset session store between tests."""
        import sous_bot.api.main as api_mod
        api_mod.store = api_mod.SessionStore()

    @patch("sous_bot.api.main.planner_engine")
    def test_chat_endpoint(self, mock_engine: MagicMock) -> None:
        from sous_bot.api.main import app
        mock_engine.chat_with_plan.return_value = (
            "Try making pasta carbonara!",
            None,
            None,
        )

        client = TestClient(app)
        resp = client.post("/chat", json={
            "message": "What should I cook?",
            "available_ingredients": ["pasta", "eggs"],
        })

        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert "carbonara" in data["message"].lower()
        assert data["plan"] is None

    @patch("sous_bot.api.main.planner_engine")
    def test_chat_with_detected_ingredients(self, mock_engine: MagicMock) -> None:
        """T3 vision detected_ingredients flow into T1 chat."""
        from sous_bot.api.main import app
        mock_engine.chat_with_plan.return_value = (
            "With parmesan and eggs, you can make carbonara!",
            None,
            None,
        )

        client = TestClient(app)
        resp = client.post("/chat", json={
            "message": "What can I make?",
            "available_ingredients": ["pasta"],
            "detected_ingredients": [
                {"name": "parmesan", "confidence": 0.95},
                {"name": "eggs", "confidence": 0.9},
            ],
        })

        assert resp.status_code == 200
        # Verify detected_ingredients were merged into inventory
        call_args = mock_engine.chat_with_plan.call_args
        inventory = call_args[0][2]  # third positional arg
        assert "parmesan" in inventory
        assert "eggs" in inventory
        assert "pasta" in inventory

    @patch("sous_bot.api.main.planner_engine")
    def test_chat_returns_plan(self, mock_engine: MagicMock) -> None:
        """Weekly request returns a plan in the response."""
        from sous_bot.api.main import app
        plan = PlanResponse(
            steps=[],
            missing_ingredients=[Ingredient(name="guanciale", quantity="1kg")],
            estimated_time="7 days",
            shopping_list=[ShoppingItem(name="guanciale", quantity="1kg", aisle="deli")],
            shopping_list_by_recipe=[
                ShoppingListByRecipe(
                    recipe="carbonara",
                    items=[ShoppingItem(name="guanciale", quantity="1kg", aisle="deli")],
                )
            ],
        )
        mock_engine.chat_with_plan.return_value = (
            "I planned carbonara for 7 days.",
            plan,
            "carbonara",
        )

        client = TestClient(app)
        resp = client.post("/chat", json={
            "session_id": "test-session",
            "message": "Make carbonara for 7 days",
            "available_ingredients": ["pasta", "eggs"],
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["plan"] is not None
        assert data["plan"]["estimated_time"] == "7 days"
        assert len(data["plan"]["shopping_list_by_recipe"]) == 1

    @patch("sous_bot.api.main.planner_engine")
    def test_shopping_list_endpoint(self, mock_engine: MagicMock) -> None:
        """GET /shopping-list returns stored plan from session."""
        from sous_bot.api.main import app
        plan = PlanResponse(
            steps=[],
            missing_ingredients=[],
            estimated_time="7 days",
            shopping_list_by_recipe=[
                ShoppingListByRecipe(
                    recipe="carbonara",
                    items=[ShoppingItem(name="guanciale", quantity="1kg", aisle="deli")],
                )
            ],
        )
        mock_engine.chat_with_plan.return_value = (
            "Planned!",
            plan,
            "carbonara",
        )

        client = TestClient(app)

        # First, create a session with a plan
        resp = client.post("/chat", json={
            "session_id": "shop-session",
            "message": "Make carbonara for 7 days",
            "available_ingredients": ["pasta"],
        })
        assert resp.status_code == 200

        # Now fetch the shopping list
        resp = client.get("/shopping-list", params={"session_id": "shop-session"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["recipes"]) == 1
        assert data["recipes"][0]["recipe"] == "carbonara"
        assert data["recipe_names"] == ["carbonara"]

    def test_shopping_list_no_session(self) -> None:
        """GET /shopping-list with unknown session returns 404."""
        from sous_bot.api.main import app
        client = TestClient(app)
        resp = client.get("/shopping-list", params={"session_id": "nonexistent"})
        assert resp.status_code == 404

    def test_session_persistence(self) -> None:
        """Multiple chat calls on same session maintain history."""
        from sous_bot.api.main import app
        import sous_bot.api.main as api_mod

        mock_engine = MagicMock()
        mock_engine.chat_with_plan.return_value = ("Reply", None, None)
        original_engine = api_mod.planner_engine
        original_store = api_mod.store
        api_mod.planner_engine = mock_engine
        api_mod.store = api_mod.SessionStore()  # fresh store

        try:
            client = TestClient(app)

            # First message — fresh session, no history
            resp1 = client.post("/chat", json={
                "session_id": "persist-fresh",
                "message": "Hello",
            })
            assert resp1.status_code == 200

            # Second message — history should include first exchange
            resp2 = client.post("/chat", json={
                "session_id": "persist-fresh",
                "message": "What about lunch?",
            })
            assert resp2.status_code == 200

            # Verify engine was called twice on the same session's history
            assert mock_engine.chat_with_plan.call_count == 2
            # Both calls share the same history list (mutable reference)
            calls = mock_engine.chat_with_plan.call_args_list
            history_ref_1 = calls[0][0][0]
            history_ref_2 = calls[1][0][0]
            assert history_ref_1 is history_ref_2  # same list object
            # After both calls, history has 4 items (2 exchanges)
            assert len(history_ref_1) == 4
            # Verify the messages in order
            assert calls[0][0][1] == "Hello"  # first message
            assert calls[1][0][1] == "What about lunch?"  # second message
        finally:
            api_mod.planner_engine = original_engine
            api_mod.store = original_store


# ── T3 → T1 full flow test ──────────────────────────────────────────


class TestT3UsesT1:
    """End-to-end: T3 vision scans → sends to T1 chat → T1 plans → T3 tracks."""

    def test_vision_to_planner_to_cart(self) -> None:
        """Full Phase 1→2→3 via API calls."""
        from sous_bot.vision.inventory import InventoryTracker
        from sous_bot.vision.routes import router as vision_router
        import sous_bot.vision.routes as routes_mod
        import sous_bot.api.main as api_mod

        # Reset T3 state
        routes_mod._tracker = InventoryTracker()
        api_mod.store = api_mod.SessionStore()

        # Build a combined app with both T1 + T3 routers
        combined_app = FastAPI()
        combined_app.include_router(vision_router)

        # Re-create T1 endpoints on the combined app with mocked engine
        mock_engine = MagicMock()

        plan = PlanResponse(
            steps=[CookingStep(step_number=1, description="Cook", duration_minutes=10)],
            missing_ingredients=[
                Ingredient(name="guanciale", quantity="150g"),
                Ingredient(name="black pepper", quantity="1 tsp"),
            ],
            estimated_time="20 minutes",
            shopping_list=[
                ShoppingItem(name="guanciale", quantity="150g", aisle="deli"),
                ShoppingItem(name="black pepper", quantity="1 tsp", aisle="spices"),
            ],
            shopping_list_by_recipe=[
                ShoppingListByRecipe(
                    recipe="carbonara",
                    items=[
                        ShoppingItem(name="guanciale", quantity="150g", aisle="deli"),
                        ShoppingItem(name="black pepper", quantity="1 tsp", aisle="spices"),
                    ],
                )
            ],
            robot_actions=[
                RobotAction(action_type="navigate", parameters={"aisle": "deli"}),
                RobotAction(action_type="grasp", parameters={"item": "guanciale"}),
            ],
        )
        mock_engine.chat_with_plan.return_value = (
            "Let's make carbonara! You need guanciale and black pepper.",
            plan,
            "carbonara",
        )

        # Mount T1's /chat endpoint with mocked engine
        original_engine = api_mod.planner_engine
        api_mod.planner_engine = mock_engine

        try:
            # Import T1's endpoint handlers after mocking
            from sous_bot.api.main import chat as t1_chat, shopping_list as t1_shopping

            combined_app.post("/chat")(t1_chat)
            combined_app.get("/shopping-list")(t1_shopping)

            client = TestClient(combined_app)

            # Phase 2: T3 scans pantry
            routes_mod._tracker.update_available(["pasta", "eggs", "parmesan"])

            # Phase 1: T3 sends detected ingredients to T1's /chat
            resp = client.post("/chat", json={
                "session_id": "demo",
                "message": "Make carbonara for 7 days",
                "available_ingredients": ["pasta", "eggs"],
                "detected_ingredients": [
                    {"name": "parmesan", "confidence": 0.95},
                ],
            })
            assert resp.status_code == 200
            chat_data = resp.json()
            assert chat_data["plan"] is not None

            # Verify T1 received detected_ingredients in inventory
            call_args = mock_engine.chat_with_plan.call_args
            inventory_passed = call_args[0][2]
            assert "parmesan" in inventory_passed
            assert "pasta" in inventory_passed

            # T3 extracts shopping list from T1's response
            shopping_items = chat_data["plan"]["shopping_list"]
            needed_names = [item["name"] for item in shopping_items]

            # T3 sets needed ingredients in inventory tracker
            resp = client.post(
                "/vision/inventory/needed",
                json={"ingredients": needed_names + ["pasta", "eggs", "parmesan"]},
            )
            assert resp.status_code == 200

            # T3 confirms missing items
            resp = client.get("/vision/shopping-list")
            assert resp.status_code == 200
            missing = resp.json()["items"]
            assert "guanciale" in missing
            assert "black pepper" in missing
            assert "pasta" not in missing

            # Phase 3: Robot fetches items → T3 tracks cart
            resp = client.post("/vision/cart/add", json={"item": "guanciale"})
            assert resp.status_code == 200
            assert resp.json()["complete"] is False

            resp = client.post("/vision/cart/add", json={"item": "black pepper"})
            assert resp.status_code == 200
            assert resp.json()["complete"] is True
            assert "Requirements met" in resp.json()["message"]

        finally:
            api_mod.planner_engine = original_engine
