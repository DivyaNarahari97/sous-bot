from __future__ import annotations

from dataclasses import dataclass, field
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from sous_bot.api.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    PlanResponse,
    ShoppingListResponse,
)
from sous_bot.planner.engine import PlannerEngine, load_settings


@dataclass
class SessionState:
    session_id: str
    history: list[ChatMessage] = field(default_factory=list)
    last_plan: PlanResponse | None = None
    last_recipe: str | None = None
    last_shopping_by_recipe: list | None = None
    last_recipe_names: list[str] | None = None


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get_or_create(self, session_id: str | None) -> SessionState:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        new_id = session_id or str(uuid.uuid4())
        state = SessionState(session_id=new_id)
        self._sessions[new_id] = state
        return state

    def get_existing(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            raise KeyError(session_id)
        return self._sessions[session_id]


settings = load_settings()
api_settings = settings.get("api", {})

app = FastAPI(title="PantryPilot API")
if api_settings.get("cors_origins"):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_settings["cors_origins"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

planner_engine = PlannerEngine()
store = SessionStore()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    state = store.get_or_create(request.session_id)
    inventory = request.available_ingredients + [item.name for item in request.detected_ingredients]

    reply, plan, recipe = planner_engine.chat_with_plan(state.history, request.message, inventory)
    state.history.append(ChatMessage(role="user", content=request.message))
    state.history.append(ChatMessage(role="assistant", content=reply))
    if plan:
        state.last_plan = plan
        state.last_recipe = recipe
        state.last_recipe_names = recipe.split(", ") if recipe else None
        state.last_shopping_by_recipe = plan.shopping_list_by_recipe

    return ChatResponse(session_id=state.session_id, message=reply, plan=state.last_plan)


@app.get("/shopping-list", response_model=ShoppingListResponse)
def shopping_list(session_id: str | None = None) -> ShoppingListResponse:
    session_id = session_id or "default"
    try:
        state = store.get_existing(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    if not state.last_plan:
        raise HTTPException(status_code=404, detail="No shopping list available")

    recipes = state.last_shopping_by_recipe
    if not recipes:
        raise HTTPException(status_code=404, detail="No shopping list available")
    return ShoppingListResponse(recipes=recipes, recipe_names=state.last_recipe_names)
