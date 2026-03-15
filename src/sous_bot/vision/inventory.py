"""Ingredient inventory state tracking and cart validation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngredientInventory(BaseModel):
    """Current state of ingredient availability."""

    available: list[str] = Field(default_factory=list)
    needed: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class CartValidation(BaseModel):
    """Result of validating the shopping cart against required items."""

    complete: bool = False
    collected: list[str] = Field(default_factory=list)
    remaining: list[str] = Field(default_factory=list)
    message: str = ""


class InventoryTracker:
    """Tracks available vs needed ingredients and computes missing."""

    def __init__(self) -> None:
        self._available: set[str] = set()
        self._needed: set[str] = set()
        self._cart: set[str] = set()

    def update_available(self, ingredients: list[str]) -> None:
        """Set the list of currently available ingredients (pantry scan)."""
        self._available = {i.lower().strip() for i in ingredients}

    def add_available(self, ingredient: str) -> None:
        """Add a single ingredient to available set."""
        self._available.add(ingredient.lower().strip())

    def set_needed(self, ingredients: list[str]) -> None:
        """Set the list of ingredients needed for recipes."""
        self._needed = {i.lower().strip() for i in ingredients}

    def get_inventory(self) -> IngredientInventory:
        """Compute and return current inventory state."""
        missing = self._needed - self._available
        return IngredientInventory(
            available=sorted(self._available),
            needed=sorted(self._needed),
            missing=sorted(missing),
        )

    # ── Cart validation (for T2 robotics shopping loop) ───────────

    def get_shopping_list(self) -> list[str]:
        """Return the list of items the robot needs to fetch."""
        return sorted(self._needed - self._available)

    def add_to_cart(self, item: str) -> CartValidation:
        """Add an item to the cart and return updated validation."""
        self._cart.add(item.lower().strip())
        return self.validate_cart()

    def validate_cart(self) -> CartValidation:
        """Check if all required items are in the cart."""
        required = self._needed - self._available
        collected = self._cart & required
        remaining = required - self._cart
        complete = len(remaining) == 0 and len(required) > 0

        if complete:
            msg = f"All {len(collected)} items collected! Requirements met."
        elif not required:
            msg = "No items needed — pantry is fully stocked."
        else:
            msg = (
                f"{len(collected)}/{len(required)} items collected. "
                f"Still need: {', '.join(sorted(remaining))}."
            )

        return CartValidation(
            complete=complete,
            collected=sorted(collected),
            remaining=sorted(remaining),
            message=msg,
        )

    def reset_cart(self) -> None:
        """Clear the cart (start a new shopping trip)."""
        self._cart.clear()
