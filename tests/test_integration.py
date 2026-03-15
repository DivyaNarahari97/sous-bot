"""Integration tests: T3 Vision/Voice ↔ T1 Planner/API.

Tests the full data flow with mocked LLM responses:
  Phase 1: User requests meal → T1 planner generates shopping list
  Phase 2: T3 vision scans pantry → detects ingredients → computes missing
  Phase 3: T2 robot fetches items → T3 cart validates → "requirements met"
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sous_bot.api.schemas import (
    DetectedIngredient as T1DetectedIngredient,
    ShoppingItem,
)
from sous_bot.vision.detector import DetectedIngredient as T3DetectedIngredient
from sous_bot.vision.inventory import InventoryTracker
from sous_bot.vision.routes import router as vision_router
from sous_bot.voice.routes import router as voice_router


def _make_mock_response(content: str) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


class TestT3T1SchemaCompat:
    """Test that T3 DetectedIngredient is compatible with T1's schema."""

    def test_t3_ingredients_convert_to_t1_format(self) -> None:
        """T3 vision output can feed into T1 planner's expected input."""
        # T3 produces these from vision scan
        t3_results = [
            T3DetectedIngredient(name="pasta", confidence=0.96),
            T3DetectedIngredient(name="eggs", confidence=0.91),
            T3DetectedIngredient(name="parmesan", confidence=0.88),
        ]

        # Convert to T1's format (which accepts name + optional confidence)
        t1_ingredients = [
            T1DetectedIngredient(name=i.name, confidence=i.confidence)
            for i in t3_results
        ]

        assert len(t1_ingredients) == 3
        assert t1_ingredients[0].name == "pasta"
        assert t1_ingredients[0].confidence == 0.96

    def test_t1_shopping_items_feed_t3_inventory(self) -> None:
        """T1 shopping list items can be used with T3 inventory tracker."""
        # T1 planner produces shopping list
        shopping_list = [
            ShoppingItem(name="guanciale", quantity="150g", aisle="deli"),
            ShoppingItem(name="black pepper", quantity="1 tsp", aisle="spices"),
            ShoppingItem(name="olive oil", quantity="1 bottle", aisle="produce"),
        ]

        # T3 inventory tracker consumes these as "needed" items
        tracker = InventoryTracker()
        tracker.update_available(["pasta", "eggs", "parmesan"])
        tracker.set_needed(
            [item.name for item in shopping_list] + ["pasta", "eggs", "parmesan"]
        )

        inv = tracker.get_inventory()
        assert "guanciale" in inv.missing
        assert "black pepper" in inv.missing
        assert "olive oil" in inv.missing
        assert "pasta" not in inv.missing


class TestFullShoppingFlow:
    """End-to-end: scan pantry → compute missing → robot shops → cart validates."""

    def test_full_shopping_loop(self) -> None:
        tracker = InventoryTracker()

        # Phase 2: Vision scans pantry
        detected = ["pasta", "eggs", "parmesan", "rice", "soy sauce"]
        tracker.update_available(detected)

        # T1 planner says we need these for carbonara + stir fry
        all_needed = [
            "pasta", "eggs", "parmesan", "guanciale", "black pepper",
            "olive oil", "rice", "soy sauce", "chicken", "ginger",
        ]
        tracker.set_needed(all_needed)

        # Check what's missing
        shopping_list = tracker.get_shopping_list()
        assert sorted(shopping_list) == [
            "black pepper", "chicken", "ginger", "guanciale", "olive oil",
        ]

        # Phase 3: Robot fetches items one by one
        for item in ["guanciale", "black pepper", "olive oil", "chicken"]:
            result = tracker.add_to_cart(item)
            assert not result.complete  # still missing ginger

        # Final item
        result = tracker.add_to_cart("ginger")
        assert result.complete is True
        assert result.remaining == []
        assert "Requirements met" in result.message


class TestVisionRoutesWithT1:
    """Test T3 vision routes serve data T1 needs."""

    @staticmethod
    def _make_app() -> FastAPI:
        app = FastAPI()
        app.include_router(vision_router)
        app.include_router(voice_router)
        return app

    def test_full_api_shopping_flow(self) -> None:
        """Simulate the full API flow T1/T2 would use."""
        client = TestClient(self._make_app())

        # Phase 2: T1 sets needed ingredients after meal planning
        resp = client.post(
            "/vision/inventory/needed",
            json={"ingredients": [
                "pasta", "eggs", "guanciale", "black pepper", "parmesan",
            ]},
        )
        assert resp.status_code == 200

        # T3 vision scan detects pantry items (mocked via direct scan)
        # In real flow, POST /vision/scan with image
        # For now, test via shopping-list endpoint
        resp = client.get("/vision/shopping-list")
        assert resp.status_code == 200
        items = resp.json()["items"]
        # All 5 are missing since no scan was done yet
        assert len(items) == 5

        # Phase 3: Robot picks items, reports to cart
        for item in ["pasta", "eggs", "guanciale", "black pepper"]:
            resp = client.post("/vision/cart/add", json={"item": item})
            assert resp.status_code == 200
            assert resp.json()["complete"] is False

        # Final item
        resp = client.post("/vision/cart/add", json={"item": "parmesan"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["complete"] is True
        assert "Requirements met" in data["message"]

        # Validate endpoint confirms
        resp = client.get("/vision/cart/validate")
        assert resp.status_code == 200
        assert resp.json()["complete"] is True

    def test_scan_with_mocked_detector(self) -> None:
        """Test POST /vision/scan with mocked vision LLM."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            '[{"name": "milk", "confidence": 0.95}, '
            '{"name": "eggs", "confidence": 0.9}]'
        )

        with patch("sous_bot.vision.detector.OpenAI", return_value=mock_client):
            # Need to reset the cached detector
            import sous_bot.vision.routes as routes_mod
            routes_mod._detector = None

            client = TestClient(self._make_app())
            resp = client.post(
                "/vision/scan",
                files={"file": ("pantry.jpg", b"fake-image-data", "image/jpeg")},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["ingredients"]) == 2
        assert data["ingredients"][0]["name"] == "milk"
