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


class LocateResult(BaseModel):
    """Result of locating a specific item on a store shelf."""

    found: bool = False
    item_name: str = ""
    position: str = ""  # e.g. "top-left", "middle-center", "bottom-right"
    pixel_x: int = -1  # Pixel X coordinate (center of item in image)
    pixel_y: int = -1  # Pixel Y coordinate (center of item in image)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    description: str = ""
    raw_response: str = ""


VISION_PROMPT = (
    "You are a pantry inventory assistant. Analyze this image and identify all "
    "visible food ingredients, groceries, and pantry items. "
    "IMPORTANT: List each individual product SEPARATELY. Do not combine adjacent "
    "items into one name. For example, 'beef' and 'yogurt' are two separate items, "
    "not 'beef yogurt'. Each name should be a single product. "
    "Return a JSON array "
    "where each element has 'name' (string, lowercase, one product per entry) and "
    "'confidence' (float 0-1). Only include items you can clearly identify. Example: "
    '[{"name": "milk", "confidence": 0.95}, {"name": "eggs", "confidence": 0.9}]'
)

SHELF_LOCATE_PROMPT = (
    "You are a grocery store shelf scanner for an assistive robot. "
    "The image is {width}x{height} pixels. "
    "Look at this shelf image and find the item: '{item_name}'. "
    "Return a JSON object with: "
    "'found' (bool), "
    "'pixel_x' (int, X pixel coordinate of the CENTER of the item), "
    "'pixel_y' (int, Y pixel coordinate of the CENTER of the item), "
    "'position' (string: top-left/top-center/top-right/"
    "middle-left/middle-center/middle-right/bottom-left/bottom-center/bottom-right), "
    "'confidence' (float 0-1), 'description' (brief visual description of the item). "
    'Example: {{"found": true, "pixel_x": 320, "pixel_y": 240, '
    '"position": "middle-center", "confidence": 0.92, '
    '"description": "red box of pasta on second shelf"}}'
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

    def locate_item_on_shelf(
        self, image_bytes: bytes, item_name: str,
        width: int = 640, height: int = 480,
    ) -> LocateResult:
        """Locate a specific item on a grocery shelf image (robot's eyes)."""
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{b64}"
        prompt = SHELF_LOCATE_PROMPT.format(
            item_name=item_name, width=width, height=height,
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            max_tokens=512,
        )
        raw = response.choices[0].message.content or ""
        return self._parse_locate_response(raw, item_name)

    def _parse_locate_response(self, raw: str, item_name: str) -> LocateResult:
        """Parse the shelf locate JSON response."""
        text = raw.strip()
        # Extract JSON object from possible markdown wrapping
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return LocateResult(item_name=item_name, raw_response=raw)

        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return LocateResult(item_name=item_name, raw_response=raw)

        return LocateResult(
            found=bool(data.get("found", False)),
            item_name=item_name,
            pixel_x=int(data.get("pixel_x") or -1),
            pixel_y=int(data.get("pixel_y") or -1),
            position=str(data.get("position", "")),
            confidence=float(data.get("confidence", 0.0)),
            description=str(data.get("description", "")),
            raw_response=raw,
        )

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
