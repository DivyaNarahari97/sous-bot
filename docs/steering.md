# PantryPilot — Steering Document

## Vision

PantryPilot is an assistive grocery robot for visually impaired and elderly users. It scans your kitchen, understands your meal plan, figures out what's missing, and helps you shop — powered by a Unitree G1 humanoid robot and Nebius AI infrastructure.

**Hackathon demo target:** "Scan my pantry, plan a meal, tell me what to buy, and show the robot fetching items in simulation."

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       PantryPilot System                         │
├───────────────┬──────────────────┬───────────────────────────────┤
│   PERCEIVE    │      REASON      │            ACT                │
│               │                  │                               │
│  Camera Feed  │  Meal Planner    │  MuJoCo Sim / Unitree G1     │
│      ↓        │  (Nebius LLM)    │                               │
│  Vision LLM   │      ↓           │  - Navigate grocery store     │
│  (Nebius TF   │  Shopping List   │  - Locate items on shelves    │
│   Qwen2.5-VL) │  Generator       │  - Reach & grasp products     │
│      ↓        │                  │  - Hand off to user           │
│  Pantry       │                  │                               │
│  Inventory    │                  │                               │
├───────────────┴──────────────────┴───────────────────────────────┤
│                VOICE INTERFACE (Accessibility)                    │
│           Whisper STT  ←→  TTS Engine                            │
├──────────────────────────────────────────────────────────────────┤
│                    FastAPI Backend                                │
│         REST + WebSocket endpoints for all modules               │
└──────────────────────────────────────────────────────────────────┘
```

### Components

1. **Vision Module** (`src/sous_bot/vision/`) — Camera feed → Nebius Vision LLM (Qwen2.5-VL via Token Factory) → pantry inventory. Scans shelves/counters, identifies food items, reads labels, estimates quantities.
2. **AI Planner** (`src/sous_bot/planner/`) — Nebius Token Factory LLM for meal planning + shopping list generation. Takes user dietary preferences + pantry inventory → produces meal plan, recipes, and a shopping list of missing items.
3. **Voice Interface** (`src/sous_bot/voice/`) — Whisper (STT) + TTS for hands-free accessibility. Critical for visually impaired users.
4. **Robotics / Simulation** (`src/sous_bot/robotics/`) — MuJoCo simulation of Unitree G1 navigating a grocery store, locating items, and performing reach-and-grasp. Adapter pattern for sim vs. real hardware.
5. **API / Backend** (`src/sous_bot/api/`) — FastAPI server. REST + WebSocket endpoints for the UI and inter-module communication.

---

## Hackathon Resources (Nebius)

| Resource | What We Use It For |
|----------|-------------------|
| **Nebius Token Factory — Qwen2.5-VL / LLaVA** | Vision LLM for pantry scanning (image → ingredient list) |
| **Nebius Token Factory — LLM (text)** | Meal planning, recipe reasoning, shopping list generation |
| **Nebius Compute (GPU)** | Running MuJoCo simulation, model inference if needed |
| **Whisper (local or Nebius-hosted)** | Speech-to-text for voice commands |
| **TTS engine** | Text-to-speech for reading back recipes/lists to user |

> **Important:** All LLM calls go through Nebius Token Factory (OpenAI-compatible API). Do NOT use Anthropic/OpenAI SDKs directly — use the `openai` Python SDK pointed at the Nebius Token Factory endpoint.

---

## Teammate Assignments

### Teammate 1 — AI Planner & Backend API

**Owner:** TBD
**Scope:** `src/sous_bot/planner/`, `src/sous_bot/api/`, `config/`, `tests/test_planner.py`
**Goal:** Wire up the Nebius LLM planner and expose it via FastAPI.

Key responsibilities:
- Planner prompt chain (meal plan → recipe → shopping list)
- All FastAPI endpoints (`/chat`, `/plan`, `/shopping-list`)
- Session/conversation state management
- Integration contract: accepts vision inventory as input, outputs action list for robotics

### Teammate 2 — Robotics & MuJoCo Simulation

**Owner:** TBD
**Scope:** `src/sous_bot/robotics/`, `sim/`, `scripts/run_demo.py`, `tests/test_robotics.py`
**Goal:** Build MuJoCo grocery store simulation with Unitree G1 performing navigation + reach-and-grasp.

Key responsibilities:
- MuJoCo grocery store environment setup
- G1 robot model loading and control policies
- Navigation + item location + grasp sequences
- Simulation adapter + `/robot/execute` endpoint
- Demo script showing full grocery shopping sequence

### Teammate 3 — Vision & Voice Interface

**Owner:** TBD
**Scope:** `src/sous_bot/vision/`, `src/sous_bot/voice/`, `tests/test_vision.py`
**Goal:** Build pantry scanning via Nebius Vision LLM + voice accessibility layer.

Key responsibilities:
- Camera capture pipeline (live + static image support)
- Nebius Token Factory Qwen2.5-VL integration for food/ingredient detection
- Pantry inventory tracker (available vs. needed vs. missing)
- Whisper STT + TTS integration for voice commands/feedback
- `/vision/ingredients` endpoint

---

## Repo Structure

```
sous-bot/
├── README.md
├── TASKS.md
├── CLAUDE.md                    # AI coding rules (merge conflict prevention)
├── pyproject.toml
├── uv.lock
├── docs/
│   └── steering.md              # this file
├── config/
│   └── settings.yaml            # runtime config (API keys, model params, camera settings)
├── scripts/
│   ├── run_demo.py              # end-to-end demo script
│   └── setup_env.sh             # environment setup helper
├── src/
│   └── sous_bot/
│       ├── __init__.py
│       ├── planner/
│       │   ├── __init__.py
│       │   ├── engine.py        # Nebius LLM planner core
│       │   ├── prompts.py       # prompt templates
│       │   └── search.py        # recipe search / grounding
│       ├── vision/
│       │   ├── __init__.py
│       │   ├── camera.py        # camera capture
│       │   ├── detector.py      # Nebius Qwen2.5-VL ingredient detection
│       │   └── inventory.py     # ingredient state tracker
│       ├── voice/
│       │   ├── __init__.py
│       │   ├── stt.py           # Whisper speech-to-text
│       │   └── tts.py           # text-to-speech output
│       ├── robotics/
│       │   ├── __init__.py
│       │   ├── controller.py    # action orchestrator
│       │   └── adapters/
│       │       ├── __init__.py
│       │       ├── base.py      # abstract adapter protocol
│       │       ├── simulation.py # MuJoCo sim adapter
│       │       └── real.py      # real G1 hardware adapter (stretch)
│       └── api/
│           ├── __init__.py
│           ├── main.py          # FastAPI app entry point
│           ├── routes.py        # endpoint definitions
│           └── schemas.py       # Pydantic request/response models
├── sim/
│   └── grocery_env.py           # MuJoCo grocery store environment
└── tests/
    ├── test_planner.py
    ├── test_vision.py
    └── test_robotics.py
```

---

## API Draft

### `POST /chat`
Conversational endpoint. Accepts user message (text or transcribed voice), returns assistant reply.
```json
{ "message": "What can I make with what's in my pantry?", "session_id": "abc123" }
→ { "reply": "I can see pasta, tomatoes, and garlic. How about spaghetti aglio e olio? You'd just need olive oil and chili flakes.", "actions": [] }
```

### `POST /plan`
Generate a full meal plan + shopping list.
```json
{ "recipe": "spaghetti carbonara", "servings": 2, "available_ingredients": ["spaghetti", "eggs", "parmesan"] }
→ { "steps": [...], "missing_ingredients": ["guanciale", "black pepper"], "estimated_time": "25 min" }
```

### `GET /shopping-list?session_id=abc123`
Returns the shopping list derived from the current plan.
```json
→ { "items": [{"name": "guanciale", "quantity": "150g", "aisle": "deli"}, {"name": "black pepper", "quantity": "1 tsp", "aisle": "spices"}] }
```

### `POST /robot/execute`
Send a grocery-fetching action to the robotics controller.
```json
{ "action": "fetch", "target": "guanciale", "location": "deli_aisle" }
→ { "status": "completed", "adapter": "simulation" }
```

### `GET /vision/ingredients`
Returns currently detected pantry ingredients.
```json
→ { "ingredients": [{"name": "spaghetti", "confidence": 0.96}, {"name": "eggs", "confidence": 0.91}] }
```

### `POST /voice/transcribe`
Transcribe voice input to text.
```json
(multipart audio file)
→ { "text": "What can I cook tonight?", "confidence": 0.93 }
```

---

## AI-Assistant Coding Rules

> See `CLAUDE.md` at the repo root for the full AI coding rules including merge conflict prevention.

1. **Use Nebius Token Factory as the LLM provider.** Use the `openai` Python SDK pointed at the Nebius Token Factory endpoint. Do NOT use the `anthropic` SDK for application code.
2. **All LLM prompts live in `src/sous_bot/planner/prompts.py`.** No inline prompt strings in business logic.
3. **Type everything.** Use Pydantic models for API schemas and dataclasses/TypedDict for internal data.
4. **Adapter pattern for robotics.** Never call hardware directly — always go through the adapter interface.
5. **Config over code.** API keys, model names, Nebius endpoints, thresholds → `config/settings.yaml` or `.env`. Never hardcode secrets.
6. **Test the planner with mocked LLM responses.** Don't burn API credits in CI.
7. **Keep functions small.** If a function exceeds 40 lines, split it.
8. **Use `uv` for dependency management.** All deps are in `pyproject.toml`.
9. **Strict file ownership.** Each teammate only modifies files in their assigned scope. See CLAUDE.md for details.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| MuJoCo sim too complex for hackathon | Demo blocked | Start with simple env; pre-record fallback video |
| Nebius Token Factory rate limits | Planner/vision fails | Cache responses; have pre-computed fallback plans |
| Vision LLM accuracy on pantry images | Wrong ingredients detected | Use high-confidence threshold; allow manual override |
| Voice recognition in noisy environment | Bad transcription | Fall back to text input; show transcription for confirmation |
| Integration between modules | Last-minute breakage | Define interfaces early; test with mocks; strict file ownership |
| Merge conflicts from parallel work | Lost time debugging | Strict file ownership per teammate; AI audit on commit |

---

## Demo Flow

**Scenario: "Scan my pantry, plan dinner, and show the robot shopping"**

1. User speaks: "What can I make for dinner?" → Whisper transcribes
2. Vision module scans pantry via camera → Nebius Qwen2.5-VL identifies: pasta, tomatoes, garlic, eggs, parmesan
3. Planner calls Nebius LLM → generates meal plan (carbonara) + identifies missing: guanciale, black pepper, olive oil
4. TTS reads back: "You have the basics for carbonara! You need guanciale, black pepper, and olive oil."
5. User says "Show me the shopping" → MuJoCo sim shows G1 robot navigating grocery store
6. Robot locates items on shelves → reach-and-grasp → places in cart
7. Shopping list displayed/read aloud with aisle locations
8. User can ask follow-up questions via voice or text

**Demo safety net:** If any module fails, the planner gracefully falls back to text-only mode (no vision, no robot — just meal plan + shopping list via chat).
