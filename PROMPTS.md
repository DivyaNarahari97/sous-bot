# Prompts & AI Coding Rules

## AI-Assistant Coding Rules

These rules apply to all AI-assisted coding on this project (Claude Code, Copilot, etc.):

1. **Claude is the primary LLM.** Use the `anthropic` Python SDK for the planner. Use `openai` only if a specific sub-task requires GPT-based tooling.
2. **Prompts live in `src/sous_bot/planner/prompts.py`.** No inline prompt strings scattered through business logic.
3. **Type everything.** Pydantic models for API schemas. Dataclasses or TypedDict for internal data structures.
4. **Adapter pattern for robotics.** Never call hardware directly — always go through the adapter interface in `src/sous_bot/robotics/adapters/base.py`.
5. **Config over code.** API keys, model names, thresholds go in `config/settings.yaml` or `.env`. Never hardcode secrets.
6. **Test with mocked LLM responses.** Don't burn API credits in tests. Mock the Anthropic client.
7. **Keep functions small.** Max ~40 lines. If longer, split it.
8. **Use `uv` for deps.** All dependencies managed via `pyproject.toml` + `uv.lock`.
9. **Graceful degradation.** If vision fails → fall back to manual input. If robot fails → fall back to text instructions. Always have a working text-only path.
10. **No hardcoded recipes.** All recipe knowledge comes from the LLM or Tavily search, not from local JSON files.

---

## Prompt Templates

All prompts below should be implemented in `src/sous_bot/planner/prompts.py` as string constants or template functions.

### SYSTEM_PROMPT — Base Planner Persona

```
You are Sous-Bot, an expert AI kitchen assistant. You help users plan meals,
generate recipes, create shopping lists, and guide cooking step-by-step.

Rules:
- Always consider dietary restrictions and allergies when mentioned.
- Provide precise quantities and timing for each step.
- When generating a shopping list, exclude ingredients the user already has.
- If unsure about an ingredient substitution, say so rather than guessing.
- Keep instructions concise but complete. Assume the user is a home cook,
  not a professional chef.
- Format steps as numbered lists. Include prep time and cook time.
```

### RECIPE_PLAN_PROMPT — Generate a Cooking Plan

```
Given the following information, generate a detailed cooking plan:

Recipe: {recipe_name}
Servings: {servings}
Available ingredients: {available_ingredients}
Dietary restrictions: {dietary_restrictions}

Return a JSON object with:
- "title": recipe name
- "servings": number
- "prep_time": estimated prep time
- "cook_time": estimated cook time
- "steps": list of objects with "step_number", "instruction", "duration", "robot_action" (optional)
- "missing_ingredients": list of objects with "name", "quantity", "unit"
- "substitutions": list of possible substitutions for missing items (if any)
```

### SHOPPING_LIST_PROMPT — Generate Shopping List

```
Based on this cooking plan, generate a shopping list.

Plan: {plan_json}
Already available: {available_ingredients}

Return a JSON object with:
- "items": list of objects with "name", "quantity", "unit", "category" (produce/dairy/meat/pantry/spice)
- "estimated_cost_range": approximate cost range in USD
- "store_sections": items grouped by store section for efficient shopping
```

### CHAT_PROMPT — Conversational Follow-Up

```
You are in a conversation with the user about cooking. Here is the context:

Current plan: {current_plan_summary}
Detected ingredients: {detected_ingredients}
Conversation history: {history}

User message: {user_message}

Respond naturally. If the user asks to modify the plan, adjust it. If they ask
a question, answer it. If they want to start cooking, confirm the plan and
indicate which robot actions will be triggered.
```

### INGREDIENT_MATCH_PROMPT — Match Vision Output to Recipe

```
The camera detected these items on the counter:
{detected_items_with_confidence}

The recipe requires:
{required_ingredients}

Match detected items to required ingredients. For each required ingredient,
indicate whether it's:
- "available": matched with a detected item (include confidence)
- "missing": not detected
- "uncertain": low-confidence match (below 0.7)

Return a JSON object with the matching results.
```

---

## Prompt Engineering Guidelines

When modifying or adding prompts:

1. **Always request structured output (JSON)** from the LLM for any data that will be parsed programmatically.
2. **Include examples** in prompts if the LLM output format is complex.
3. **Set temperature low (0.2-0.4)** for planning/structured tasks, higher (0.7) for conversational responses.
4. **Use system prompts** for persona/rules, user prompts for the specific request.
5. **Version your prompts.** When you change a prompt, note what changed and why in a comment.
6. **Test prompt changes** against at least 3 different recipe types (simple, complex, dietary-restricted) before merging.
