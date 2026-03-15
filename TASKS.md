# Sous Bot — Tasks

## Teammate 1 — AI Planner & Backend API

**Scope:** `src/sous_bot/planner/`, `src/sous_bot/api/`, `config/`, `tests/test_planner.py`
**Focus:** Nebius LLM-powered meal planner + FastAPI backend.

### Tasks

- [ ] **P1-1:** Set up FastAPI app skeleton in `src/sous_bot/api/main.py`
- [ ] **P1-2:** Define Pydantic request/response schemas in `src/sous_bot/api/schemas.py`
  - `ChatRequest`, `ChatResponse`, `PlanRequest`, `PlanResponse`, `ShoppingListResponse`
- [ ] **P1-3:** Configure Nebius Token Factory client in `src/sous_bot/planner/engine.py`
  - Use `openai` SDK with Nebius TF base URL + API key from `config/settings.yaml`
- [ ] **P1-4:** Build the planner prompt chain in `src/sous_bot/planner/engine.py`
  - Pantry inventory → meal suggestions → recipe steps → shopping list
  - All prompts stored in `src/sous_bot/planner/prompts.py`
- [ ] **P1-5:** Implement recipe search/grounding in `src/sous_bot/planner/search.py`
  - Use Tavily or web search to ground LLM suggestions in real recipes
- [ ] **P1-6:** Implement `POST /chat` endpoint — conversational, multi-turn
- [ ] **P1-7:** Implement `POST /plan` endpoint — full meal plan generation
- [ ] **P1-8:** Implement `GET /shopping-list` endpoint — derived from current plan
- [ ] **P1-9:** Add session/conversation state management
- [ ] **P1-10:** Define integration contracts:
  - Accept `list[DetectedIngredient]` from vision module as available ingredients
  - Output `list[RobotAction]` for robotics controller to execute
- [ ] **P1-11:** Write tests with mocked LLM responses (`tests/test_planner.py`)

### Key Interfaces (you own these)

```python
# Planner input
PlanRequest(recipe: str, servings: int, available_ingredients: list[str])

# Planner output
PlanResponse(steps: list[CookingStep], missing_ingredients: list[Ingredient], estimated_time: str)

# Shopping list item
ShoppingItem(name: str, quantity: str, aisle: str | None)
```

### DO NOT TOUCH

- `src/sous_bot/vision/` — owned by Teammate 3
- `src/sous_bot/voice/` — owned by Teammate 3
- `src/sous_bot/robotics/` — owned by Teammate 2
- `sim/` — owned by Teammate 2

---

## Teammate 2 — Robotics & MuJoCo Simulation

**Scope:** `src/sous_bot/robotics/`, `sim/`, `scripts/run_demo.py`, `tests/test_robotics.py`
**Focus:** MuJoCo grocery store simulation with Unitree G1 robot.

### Tasks

- [ ] **R2-1:** Define `RobotAdapter` abstract base in `src/sous_bot/robotics/adapters/base.py`
  - Methods: `navigate()`, `locate_item()`, `reach()`, `grasp()`, `place_in_cart()`, `hand_off()`, `status()`
- [ ] **R2-2:** Set up MuJoCo grocery store environment in `sim/grocery_env.py`
  - Shelves with items, aisles, cart, checkout area
  - Load Unitree G1 model
- [ ] **R2-3:** Implement `SimulationAdapter` in `src/sous_bot/robotics/adapters/simulation.py`
  - Connects to MuJoCo env, executes navigation + reach-and-grasp policies
  - Logs actions, returns status with visualization state
- [ ] **R2-4:** Implement navigation policy — G1 moves to target aisle/shelf
- [ ] **R2-5:** Implement reach-and-grasp policy — G1 picks item from shelf, places in cart
- [ ] **R2-6:** Implement `RobotController` orchestrator in `src/sous_bot/robotics/controller.py`
  - Takes a shopping list → plans route through store → executes fetch sequence
- [ ] **R2-7:** Implement `POST /robot/execute` API route (coordinate with Teammate 1 on schema only)
- [ ] **R2-8:** Create demo script `scripts/run_demo.py` — full grocery shopping sequence in sim
- [ ] **R2-9:** Stub out `RealAdapter` in `src/sous_bot/robotics/adapters/real.py` (stretch goal)
- [ ] **R2-10:** Write tests for controller + sim adapter (`tests/test_robotics.py`)

### Key Interfaces (you own these)

```python
# Robot action
class RobotAction:
    action: Literal["navigate", "locate", "reach", "grasp", "place_in_cart", "hand_off"]
    target: str
    parameters: dict  # aisle, shelf_position, etc.

# Adapter protocol
class RobotAdapter(Protocol):
    async def execute(self, action: RobotAction) -> ActionResult: ...
    async def status(self) -> AdapterStatus: ...
```

### DO NOT TOUCH

- `src/sous_bot/planner/` — owned by Teammate 1
- `src/sous_bot/api/` (except coordinating on `/robot/execute` schema) — owned by Teammate 1
- `src/sous_bot/vision/` — owned by Teammate 3
- `src/sous_bot/voice/` — owned by Teammate 3
- `config/` — owned by Teammate 1

---

## Teammate 3 — Vision & Voice Interface

**Scope:** `src/sous_bot/vision/`, `src/sous_bot/voice/`, `tests/test_vision.py`
**Focus:** Pantry scanning via Nebius Vision LLM + voice accessibility.

### Tasks

#### Vision
- [ ] **V3-1:** Set up camera capture in `src/sous_bot/vision/camera.py`
  - Support live camera + static image input (for testing/demo)
- [ ] **V3-2:** Integrate Nebius Token Factory Qwen2.5-VL in `src/sous_bot/vision/detector.py`
  - Send pantry/shelf images → get back ingredient list with confidence scores
  - Use `openai` SDK with Nebius TF endpoint (vision model)
- [ ] **V3-3:** Build ingredient inventory tracker in `src/sous_bot/vision/inventory.py`
  - Tracks what's currently visible vs. what the plan needs
  - Exposes `available`, `needed`, `missing` lists
- [ ] **V3-4:** Implement `GET /vision/ingredients` API route (coordinate with Teammate 1 on schema only)
- [ ] **V3-5:** Create test images/fixtures for demo reliability
- [ ] **V3-6:** Write tests with sample images (`tests/test_vision.py`)

#### Voice
- [ ] **V3-7:** Implement Whisper STT in `src/sous_bot/voice/stt.py`
  - Accept audio input → return transcribed text
  - Handle microphone capture or audio file input
- [ ] **V3-8:** Implement TTS in `src/sous_bot/voice/tts.py`
  - Accept text → produce audio output
  - Support reading recipes, shopping lists, and status updates aloud
- [ ] **V3-9:** Implement `POST /voice/transcribe` API route (coordinate with Teammate 1 on schema only)
- [ ] **V3-10:** Wire voice → planner pipeline (voice command → transcribe → send to /chat → TTS response)

### Key Interfaces (you own these)

```python
# Detection output
class DetectedIngredient:
    name: str
    confidence: float
    bounding_box: tuple[int, int, int, int] | None

# Inventory state
class IngredientInventory:
    available: list[DetectedIngredient]
    needed: list[str]       # from plan
    missing: list[str]      # needed - available

# Voice transcription
class TranscriptionResult:
    text: str
    confidence: float
    language: str
```

### DO NOT TOUCH

- `src/sous_bot/planner/` — owned by Teammate 1
- `src/sous_bot/api/` (except coordinating on vision/voice endpoint schemas) — owned by Teammate 1
- `src/sous_bot/robotics/` — owned by Teammate 2
- `sim/` — owned by Teammate 2
- `config/` — owned by Teammate 1

---

## Integration Milestones

| Milestone | Involves | Description |
|-----------|----------|-------------|
| **M1** | T1 | Planner returns meal plan + shopping list via `/chat` (text-only, no vision) |
| **M2** | T3 | Vision LLM detects 3+ ingredients from a pantry image |
| **M3** | T2 | MuJoCo env loads with G1 robot, basic navigation works |
| **M4** | T3 | Voice input transcribed and TTS reads back a response |
| **M5** | T1 + T3 | Planner uses vision output to auto-fill available ingredients |
| **M6** | T1 + T2 | Planner sends shopping list to robotics → G1 fetches items in sim |
| **M7** | T3 + T1 | Voice-driven end-to-end: speak → plan → shopping list read aloud |
| **M8** | All | Full demo: voice → vision → plan → robot sim → shopping list → TTS |

**Priority order:** M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8

---

## Coordination Rules

1. **Each teammate works ONLY in their assigned scope.** See "DO NOT TOUCH" sections above.
2. **Interface changes require agreement.** If you need to change a shared data type (e.g., `DetectedIngredient`, `RobotAction`), notify the other teammate first.
3. **API route registration:** Teammate 1 owns `routes.py`, but T2 and T3 can propose new routes by adding them to a `routes_{module}.py` file in their own module directory, which T1 will import.
4. **Commit often, commit small.** Each commit should touch only files in your scope.
5. **Before pushing, verify no files outside your scope were modified** (see CLAUDE.md for automated audit).
