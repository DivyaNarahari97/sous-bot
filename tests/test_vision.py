"""Tests for vision module — all LLM calls are mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sous_bot.vision.camera import CameraCapture
from sous_bot.vision.detector import IngredientDetector
from sous_bot.vision.inventory import InventoryTracker


# --- Detector tests ---


def _make_mock_response(content: str) -> MagicMock:
    """Create a mock OpenAI chat completion response."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


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


# --- Camera tests ---


class TestCameraCapture:
    def test_load_image_from_file(self, tmp_path: Path) -> None:
        # Create a minimal valid image file
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
