# CLAUDE.md — AI Assistant Rules for PantryPilot

## Project Overview

PantryPilot: assistive grocery robot for visually impaired/elderly users. Built at Nebius.Build SF Hackathon.
See `docs/steering.md` for full architecture and `TASKS.md` for task assignments.

## Tech Stack

- **LLM Provider:** Nebius Token Factory (OpenAI-compatible API). Use `openai` Python SDK pointed at Nebius endpoint.
- **Vision LLM:** Nebius Token Factory Qwen2.5-VL / LLaVA
- **Voice:** Whisper (STT) + TTS
- **Simulation:** MuJoCo + Unitree G1
- **Backend:** Python, FastAPI
- **Package Manager:** `uv` (deps in `pyproject.toml`)

## Critical Rule: File Ownership & Merge Conflict Prevention

Each teammate owns specific directories. **AI assistants MUST NOT modify files outside the current teammate's scope.**

### Ownership Map

| Teammate | Owned Paths | Description |
|----------|------------|-------------|
| **T1 — Planner/API** | `src/sous_bot/planner/`, `src/sous_bot/api/`, `config/`, `tests/test_planner.py` | AI planner + backend |
| **T2 — Robotics** | `src/sous_bot/robotics/`, `sim/`, `scripts/run_demo.py`, `tests/test_robotics.py` | MuJoCo sim + robot control |
| **T3 — Vision/Voice** | `src/sous_bot/vision/`, `src/sous_bot/voice/`, `tests/test_vision.py` | Pantry scanning + accessibility |
| **Shared (read-only)** | `README.md`, `TASKS.md`, `docs/steering.md`, `CLAUDE.md`, `pyproject.toml` | Only modify with team agreement |

### Rules for AI Assistants

1. **ASK which teammate you are working for** at the start of each session if not clear from context.
2. **NEVER create, edit, or delete files outside the current teammate's owned paths.**
3. **NEVER modify shared files** (`README.md`, `TASKS.md`, `docs/steering.md`, `CLAUDE.md`, `pyproject.toml`) without explicit user confirmation that the whole team agrees.
4. **If you need to change an interface** that another teammate depends on (e.g., a Pydantic schema, a protocol class), STOP and ask the user to coordinate with the other teammate first.
5. **If a task requires touching another teammate's code**, output the suggested change as a comment/suggestion — do NOT apply it.

## Pre-Commit Audit Rules

**Before every commit, the AI assistant MUST perform this audit:**

1. Run `git diff --name-only` to list all changed files.
2. Verify EVERY changed file is within the current teammate's owned paths (see Ownership Map above).
3. If ANY file is outside scope:
   - **DO NOT commit.**
   - Warn the user: "File X is outside your scope (owned by Teammate Y). Remove it from this commit or coordinate with them."
   - Use `git checkout -- <file>` to revert out-of-scope changes (only after user confirms).
4. Check that no new files were created outside the owned paths.
5. Check that `pyproject.toml` and `uv.lock` are only modified if the user explicitly asked to add a dependency.

### Audit Checklist (run mentally before every commit)

```
[ ] All modified files are within my teammate's scope
[ ] No files in other teammate's directories were touched
[ ] No shared files modified without team agreement
[ ] No hardcoded API keys, secrets, or tokens in the diff
[ ] No import of modules that don't exist yet (unless creating them in-scope)
[ ] Commit message clearly states which module was changed
```

## Coding Standards

1. **Nebius Token Factory for all LLM calls.** Use `openai` SDK with Nebius base URL. Do NOT use `anthropic` SDK in application code.
2. **All LLM prompts in `src/sous_bot/planner/prompts.py`.** No inline prompt strings.
3. **Type everything.** Pydantic for API schemas, dataclasses/TypedDict for internal data.
4. **Adapter pattern for robotics.** Never call hardware directly.
5. **Config over code.** API keys, model names, endpoints → `config/settings.yaml` or `.env`. Never hardcode.
6. **Test with mocked LLM responses.** No real API calls in tests.
7. **Keep functions under 40 lines.**
8. **Use `uv` for deps.** `uv add <package>` to add, `uv sync` to install.

## Commit Message Format

```
[module] short description

Examples:
[planner] add meal plan generation prompt chain
[api] implement /chat endpoint with session state
[robotics] set up MuJoCo grocery store environment
[vision] integrate Qwen2.5-VL for pantry scanning
[voice] add Whisper STT pipeline
[sim] add G1 reach-and-grasp policy
```

## Integration Points (Shared Contracts)

These data types are shared between modules. Changes require coordination:

```python
# Vision → Planner (T3 defines, T1 consumes)
class DetectedIngredient:
    name: str
    confidence: float

# Planner → Robotics (T1 defines, T2 consumes)
class ShoppingItem:
    name: str
    quantity: str
    aisle: str | None

# Planner → Robotics (T1 defines action list, T2 executes)
class RobotAction:
    action: str  # "navigate", "locate", "reach", "grasp", "place_in_cart", "hand_off"
    target: str
    parameters: dict
```

If you need to change any of these, STOP and coordinate with the other teammate.
