"""Ingredient detection using Nebius Token Factory Qwen2.5-VL vision model."""

from __future__ import annotations

import base64
import json
import os

from openai import OpenAI
from pydantic import BaseModel, Field


class DetectedIngredient(BaseModel):
    """A single detected ingredient from a pantry image."""

    name: str
    confidence: float = Field(ge=0.0, le=1.0)


class DetectionResult(BaseModel):
    """Result of running ingredient detection on an image."""

    ingredients: list[DetectedIngredient] = Field(default_factory=list)
    raw_response: str = ""


VISION_PROMPT = (
    "You are a pantry inventory assistant. Analyze this image and identify all "
    "visible food ingredients, groceries, and pantry items. Return a JSON array "
    "where each element has 'name' (string, lowercase) and 'confidence' (float "
    "0-1). Only include items you can clearly identify. Example: "
    '[{"name": "milk", "confidence": 0.95}, {"name": "eggs", "confidence": 0.9}]'
)

DEFAULT_CONFIDENCE_THRESHOLD = 0.7


class IngredientDetector:
    """Detects ingredients in images using Nebius Token Factory vision LLM."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=api_key or os.environ["NEBIUS_API_KEY"],
        )
        self._model = model
        self._confidence_threshold = confidence_threshold

    def detect_from_bytes(self, image_bytes: bytes) -> DetectionResult:
        """Detect ingredients from raw JPEG/PNG bytes."""
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return self._call_vision(f"data:image/jpeg;base64,{b64}")

    def detect_from_file(self, path: str) -> DetectionResult:
        """Detect ingredients from an image file path."""
        with open(path, "rb") as f:
            return self.detect_from_bytes(f.read())

    def _call_vision(self, image_url: str) -> DetectionResult:
        """Send image to vision LLM and parse response."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            max_tokens=1024,
        )
        raw = response.choices[0].message.content or ""
        ingredients = self._parse_response(raw)
        return DetectionResult(ingredients=ingredients, raw_response=raw)

    def _parse_response(self, raw: str) -> list[DetectedIngredient]:
        """Parse JSON response and filter by confidence threshold."""
        # Extract JSON array from response (may be wrapped in markdown)
        text = raw.strip()
        if "```" in text:
            start = text.index("[")
            end = text.rindex("]") + 1
            text = text[start:end]
        elif text.startswith("["):
            pass
        else:
            # Try to find JSON array in the text
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                text = text[start : end + 1]
            else:
                return []

        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            return []

        results: list[DetectedIngredient] = []
        for item in items:
            if isinstance(item, dict) and "name" in item and "confidence" in item:
                ingredient = DetectedIngredient(
                    name=str(item["name"]).lower(),
                    confidence=float(item["confidence"]),
                )
                if ingredient.confidence >= self._confidence_threshold:
                    results.append(ingredient)
        return results
