"""Ingredient inventory state tracking."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngredientInventory(BaseModel):
    """Current state of ingredient availability."""

    available: list[str] = Field(default_factory=list)
    needed: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class InventoryTracker:
    """Tracks available vs needed ingredients and computes missing."""

    def __init__(self) -> None:
        self._available: set[str] = set()
        self._needed: set[str] = set()

    def update_available(self, ingredients: list[str]) -> None:
        """Set the list of currently available ingredients."""
        self._available = {i.lower().strip() for i in ingredients}

    def add_available(self, ingredient: str) -> None:
        """Add a single ingredient to available set."""
        self._available.add(ingredient.lower().strip())

    def set_needed(self, ingredients: list[str]) -> None:
        """Set the list of ingredients needed for a recipe."""
        self._needed = {i.lower().strip() for i in ingredients}

    def get_inventory(self) -> IngredientInventory:
        """Compute and return current inventory state."""
        missing = self._needed - self._available
        return IngredientInventory(
            available=sorted(self._available),
            needed=sorted(self._needed),
            missing=sorted(missing),
        )
