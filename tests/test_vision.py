"""Tests for vision module — all LLM calls are mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sous_bot.vision.camera import CameraCapture
from sous_bot.vision.detector import IngredientDetector
from sous_bot.vision.inventory import InventoryTracker


# --- Helpers ---


def _make_mock_response(content: str) -> MagicMock:
    """Create a mock OpenAI chat completion response."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


# --- Detector tests ---


class TestIngredientDetector:
    def test_detect_parses_json_response(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            '[{"name": "Milk", "confidence": 0.95}, '
            '{"name": "eggs", "confidence": 0.88}]'
        )
        with patch("sous_bot.vision.detector.OpenAI", return_value=mock_client):
            detector = IngredientDetector(api_key="test-key")
            result = detector.detect_from_bytes(b"fake-image-data")

        assert len(result.ingredients) == 2
        assert result.ingredients[0].name == "milk"
        assert result.ingredients[0].confidence == 0.95
        assert result.ingredients[1].name == "eggs"

    def test_detect_filters_low_confidence(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            '[{"name": "milk", "confidence": 0.95}, '
            '{"name": "maybe_flour", "confidence": 0.3}]'
        )
        with patch("sous_bot.vision.detector.OpenAI", return_value=mock_client):
            detector = IngredientDetector(api_key="test-key")
            result = detector.detect_from_bytes(b"fake-image-data")

        assert len(result.ingredients) == 1
        assert result.ingredients[0].name == "milk"

    def test_detect_handles_markdown_wrapped_json(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            '```json\n[{"name": "butter", "confidence": 0.9}]\n```'
        )
        with patch("sous_bot.vision.detector.OpenAI", return_value=mock_client):
            detector = IngredientDetector(api_key="test-key")
            result = detector.detect_from_bytes(b"fake-image-data")

        assert len(result.ingredients) == 1
        assert result.ingredients[0].name == "butter"

    def test_detect_handles_malformed_response(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "Sorry, I cannot identify items in this image."
        )
        with patch("sous_bot.vision.detector.OpenAI", return_value=mock_client):
            detector = IngredientDetector(api_key="test-key")
            result = detector.detect_from_bytes(b"fake-image-data")

        assert len(result.ingredients) == 0
        assert "Sorry" in result.raw_response

    def test_detect_custom_confidence_threshold(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            '[{"name": "milk", "confidence": 0.85}, '
            '{"name": "flour", "confidence": 0.92}]'
        )
        with patch("sous_bot.vision.detector.OpenAI", return_value=mock_client):
            detector = IngredientDetector(
                api_key="test-key", confidence_threshold=0.9
            )
            result = detector.detect_from_bytes(b"fake-image-data")

        assert len(result.ingredients) == 1
        assert result.ingredients[0].name == "flour"


# --- Shelf locator tests ---


class TestShelfLocator:
    def test_locate_item_found(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            '{"found": true, "position": "middle-left", '
            '"confidence": 0.92, "description": "red box of pasta"}'
        )
        with patch("sous_bot.vision.detector.OpenAI", return_value=mock_client):
            detector = IngredientDetector(api_key="test-key")
            result = detector.locate_item_on_shelf(b"fake-shelf-image", "pasta")

        assert result.found is True
        assert result.item_name == "pasta"
        assert result.position == "middle-left"
        assert result.confidence == 0.92
        assert "pasta" in result.description

    def test_locate_item_not_found(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            '{"found": false, "position": "", '
            '"confidence": 0.0, "description": "item not visible on shelf"}'
        )
        with patch("sous_bot.vision.detector.OpenAI", return_value=mock_client):
            detector = IngredientDetector(api_key="test-key")
            result = detector.locate_item_on_shelf(b"fake-shelf-image", "guanciale")

        assert result.found is False
        assert result.item_name == "guanciale"

    def test_locate_handles_markdown_wrapped(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            '```json\n{"found": true, "position": "top-right", '
            '"confidence": 0.88, "description": "bottle of olive oil"}\n```'
        )
        with patch("sous_bot.vision.detector.OpenAI", return_value=mock_client):
            detector = IngredientDetector(api_key="test-key")
            result = detector.locate_item_on_shelf(b"fake-shelf-image", "olive oil")

        assert result.found is True
        assert result.position == "top-right"

    def test_locate_handles_malformed_response(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "I cannot see that item clearly."
        )
        with patch("sous_bot.vision.detector.OpenAI", return_value=mock_client):
            detector = IngredientDetector(api_key="test-key")
            result = detector.locate_item_on_shelf(b"fake-shelf-image", "milk")

        assert result.found is False
        assert result.item_name == "milk"


# --- Inventory tracker tests ---


class TestInventoryTracker:
    def test_missing_computation(self) -> None:
        tracker = InventoryTracker()
        tracker.update_available(["milk", "eggs", "butter"])
        tracker.set_needed(["milk", "flour", "sugar", "eggs"])
        inv = tracker.get_inventory()

        assert inv.missing == ["flour", "sugar"]
        assert "milk" in inv.available
        assert "eggs" in inv.available

    def test_nothing_missing(self) -> None:
        tracker = InventoryTracker()
        tracker.update_available(["milk", "eggs"])
        tracker.set_needed(["milk", "eggs"])
        inv = tracker.get_inventory()

        assert inv.missing == []

    def test_all_missing(self) -> None:
        tracker = InventoryTracker()
        tracker.update_available([])
        tracker.set_needed(["flour", "sugar"])
        inv = tracker.get_inventory()

        assert inv.missing == ["flour", "sugar"]

    def test_add_available(self) -> None:
        tracker = InventoryTracker()
        tracker.set_needed(["milk", "eggs"])
        tracker.add_available("milk")
        inv = tracker.get_inventory()

        assert inv.missing == ["eggs"]

    def test_case_insensitive(self) -> None:
        tracker = InventoryTracker()
        tracker.update_available(["Milk", "EGGS"])
        tracker.set_needed(["milk", "eggs"])
        inv = tracker.get_inventory()

        assert inv.missing == []


# --- Cart validation tests ---


class TestCartValidation:
    def test_cart_add_and_validate(self) -> None:
        tracker = InventoryTracker()
        tracker.update_available(["pasta", "eggs"])
        tracker.set_needed(["pasta", "eggs", "guanciale", "black pepper"])

        # Robot picks up guanciale
        result = tracker.add_to_cart("guanciale")
        assert not result.complete
        assert "guanciale" in result.collected
        assert "black pepper" in result.remaining
        assert "1/2" in result.message

    def test_cart_complete(self) -> None:
        tracker = InventoryTracker()
        tracker.update_available(["pasta"])
        tracker.set_needed(["pasta", "milk", "butter"])

        tracker.add_to_cart("milk")
        result = tracker.add_to_cart("butter")
        assert result.complete is True
        assert result.remaining == []
        assert "Requirements met" in result.message

    def test_cart_validate_no_items_needed(self) -> None:
        tracker = InventoryTracker()
        tracker.update_available(["milk", "eggs"])
        tracker.set_needed(["milk", "eggs"])

        result = tracker.validate_cart()
        assert not result.complete  # nothing to shop for
        assert "fully stocked" in result.message

    def test_cart_reset(self) -> None:
        tracker = InventoryTracker()
        tracker.set_needed(["milk", "eggs"])
        tracker.add_to_cart("milk")
        tracker.reset_cart()

        result = tracker.validate_cart()
        assert "milk" in result.remaining

    def test_shopping_list(self) -> None:
        tracker = InventoryTracker()
        tracker.update_available(["pasta", "rice"])
        tracker.set_needed(["pasta", "rice", "chicken", "soy sauce"])

        items = tracker.get_shopping_list()
        assert items == ["chicken", "soy sauce"]


# --- Camera tests ---


class TestCameraCapture:
    def test_load_image_from_file(self, tmp_path: Path) -> None:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        import cv2

        img_path = tmp_path / "test.jpg"
        cv2.imwrite(str(img_path), img)

        result = CameraCapture.load_image(img_path)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_load_image_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            CameraCapture.load_image("/nonexistent/image.jpg")


# --- FastAPI route tests ---


class TestVisionRoutes:
    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        import sous_bot.vision.routes as routes_mod
        from sous_bot.vision.routes import router

        # Reset shared state between tests
        routes_mod._tracker = InventoryTracker()

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_get_ingredients_empty(self, client) -> None:
        resp = client.get("/vision/ingredients")
        assert resp.status_code == 200
        assert resp.json()["ingredients"] == []

    def test_get_inventory(self, client) -> None:
        resp = client.get("/vision/inventory")
        assert resp.status_code == 200
        data = resp.json()
        assert "available" in data
        assert "needed" in data
        assert "missing" in data

    def test_set_needed(self, client) -> None:
        resp = client.post(
            "/vision/inventory/needed",
            json={"ingredients": ["milk", "eggs", "flour"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "milk" in data["needed"]

    def test_cart_add_and_validate(self, client) -> None:
        # Set up needed items
        client.post(
            "/vision/inventory/needed",
            json={"ingredients": ["chicken"]},
        )
        # Add to cart
        resp = client.post("/vision/cart/add", json={"item": "chicken"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["complete"] is True

    def test_cart_reset(self, client) -> None:
        resp = client.post("/vision/cart/reset")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_scan_empty_file(self, client) -> None:
        resp = client.post(
            "/vision/scan",
            files={"file": ("test.jpg", b"", "image/jpeg")},
        )
        assert resp.status_code == 400
