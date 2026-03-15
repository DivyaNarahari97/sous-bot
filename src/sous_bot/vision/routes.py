"""T3 Vision FastAPI routes — T1 imports these into the main app."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from sous_bot.vision.camera import CameraCapture
from sous_bot.vision.detector import (
    DetectedIngredient,
    DetectionResult,
    IngredientDetector,
    LocateResult,
)
from sous_bot.vision.inventory import CartValidation, InventoryTracker

load_dotenv()

router = APIRouter(prefix="/vision", tags=["vision"])

# Shared state — singleton instances
_detector: IngredientDetector | None = None
_tracker = InventoryTracker()


def _get_detector() -> IngredientDetector:
    global _detector
    if _detector is None:
        _detector = IngredientDetector()
    return _detector


# ── Request/Response schemas ──────────────────────────────────────


class IngredientsResponse(BaseModel):
    ingredients: list[DetectedIngredient] = Field(default_factory=list)
    raw_response: str = ""


class LocateRequest(BaseModel):
    item_name: str


class InventoryResponse(BaseModel):
    available: list[str] = Field(default_factory=list)
    needed: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class SetNeededRequest(BaseModel):
    ingredients: list[str]


class CartAddRequest(BaseModel):
    item: str


# ── Endpoints ─────────────────────────────────────────────────────


@router.get("/ingredients", response_model=IngredientsResponse)
async def get_ingredients():
    """Return currently detected pantry ingredients from last scan."""
    inv = _tracker.get_inventory()
    return IngredientsResponse(
        ingredients=[
            DetectedIngredient(name=name, confidence=1.0)
            for name in inv.available
        ]
    )


@router.post("/scan", response_model=IngredientsResponse)
async def scan_image(file: UploadFile = File(...)):
    """Scan an uploaded image for ingredients using Nebius Qwen2.5-VL."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    detector = _get_detector()
    result = detector.detect_from_bytes(contents)

    # Update tracker with detected ingredients
    names = [i.name for i in result.ingredients]
    _tracker.update_available(names)

    return IngredientsResponse(
        ingredients=result.ingredients,
        raw_response=result.raw_response,
    )


@router.post("/scan-camera", response_model=IngredientsResponse)
async def scan_camera():
    """Capture from webcam and scan for ingredients."""
    try:
        camera = CameraCapture()
        image_bytes = camera.capture_frame()
        camera.release()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Camera error: {e}")

    detector = _get_detector()
    result = detector.detect_from_bytes(image_bytes)
    names = [i.name for i in result.ingredients]
    _tracker.update_available(names)

    return IngredientsResponse(
        ingredients=result.ingredients,
        raw_response=result.raw_response,
    )


@router.post("/locate-item", response_model=LocateResult)
async def locate_item(
    request: LocateRequest, file: UploadFile = File(...)
):
    """Locate a specific item on a shelf image (robot's eyes)."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    detector = _get_detector()
    return detector.locate_item_on_shelf(contents, request.item_name)


# ── Inventory / Cart endpoints (for T1 planner + T2 robotics) ────


@router.get("/inventory", response_model=InventoryResponse)
async def get_inventory():
    """Get current inventory state (available, needed, missing)."""
    inv = _tracker.get_inventory()
    return InventoryResponse(
        available=inv.available,
        needed=inv.needed,
        missing=inv.missing,
    )


@router.post("/inventory/needed", response_model=InventoryResponse)
async def set_needed(request: SetNeededRequest):
    """Set the needed ingredients list (called by T1 planner)."""
    _tracker.set_needed(request.ingredients)
    inv = _tracker.get_inventory()
    return InventoryResponse(
        available=inv.available,
        needed=inv.needed,
        missing=inv.missing,
    )


@router.get("/shopping-list")
async def get_shopping_list() -> dict:
    """Get the shopping list (missing items for T2 robot to fetch)."""
    return {"items": _tracker.get_shopping_list()}


@router.post("/cart/add", response_model=CartValidation)
async def add_to_cart(request: CartAddRequest):
    """Add item to cart (called by T2 robot after grasping)."""
    return _tracker.add_to_cart(request.item)


@router.get("/cart/validate", response_model=CartValidation)
async def validate_cart():
    """Validate if all required items are in the cart."""
    return _tracker.validate_cart()


@router.post("/cart/reset")
async def reset_cart() -> dict:
    """Reset the cart for a new shopping trip."""
    _tracker.reset_cart()
    return {"status": "ok", "message": "Cart reset"}
