from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DetectedIngredient(BaseModel):
    name: str
    quantity: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] | None = None


class Ingredient(BaseModel):
    name: str
    quantity: str | None = None


class CookingStep(BaseModel):
    step_number: int
    description: str
    duration_minutes: int | None = None
    tools: list[str] | None = None


class ShoppingItem(BaseModel):
    name: str
    quantity: str
    aisle: str | None = None


class RobotAction(BaseModel):
    action_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    available_ingredients: list[str] = Field(default_factory=list)
    detected_ingredients: list[DetectedIngredient] = Field(default_factory=list)


class ChatResponse(BaseModel):
    session_id: str
    message: str
    plan: PlanResponse | None = None


class PlanRequest(BaseModel):
    session_id: str | None = None
    recipe: str
    servings: int
    available_ingredients: list[str] = Field(default_factory=list)
    detected_ingredients: list[DetectedIngredient] = Field(default_factory=list)


class PlanResponse(BaseModel):
    steps: list[CookingStep]
    missing_ingredients: list[Ingredient]
    estimated_time: str
    shopping_list: list[ShoppingItem] | None = None
    robot_actions: list[RobotAction] | None = None


class ShoppingListResponse(BaseModel):
    items: list[ShoppingItem]
    recipe: str | None = None
