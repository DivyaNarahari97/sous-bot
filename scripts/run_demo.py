#!/usr/bin/env python3
"""Sous Bot — Hackathon Demo Script.

Runs the full 3-phase flow:
  Phase 1: User says what they want to cook → T1 planner generates recipe + ingredients
  Phase 2: Vision scans pantry → detects available → computes missing → shopping list
  Phase 3: Simulated robot fetches items → cart validates → "requirements met!"

Usage:
  # Full demo with voice TTS (speaks everything aloud)
  uv run python scripts/run_demo.py

  # Silent mode (text only, no TTS)
  uv run python scripts/run_demo.py --silent

  # With a real pantry image
  uv run python scripts/run_demo.py --image path/to/pantry.jpg

  # Interactive mode (lets you type commands after the scripted demo)
  uv run python scripts/run_demo.py --interactive
"""

from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

# ── Helpers ───────────────────────────────────────────────────────

SILENT = "--silent" in sys.argv
TTS_ENGINE = None


def say(text: str, pause: float = 0.5) -> None:
    """Print and optionally speak text."""
    print(f"\n🤖 Sous Bot: {text}")
    if not SILENT:
        global TTS_ENGINE
        if TTS_ENGINE is None:
            from sous_bot.voice.tts import TextToSpeech
            TTS_ENGINE = TextToSpeech()
        TTS_ENGINE.speak(text)
    time.sleep(pause)


def banner(phase: str) -> None:
    """Print a phase banner."""
    print(f"\n{'='*60}")
    print(f"  {phase}")
    print(f"{'='*60}")


def user_says(text: str) -> None:
    """Simulate user speech."""
    print(f"\n🎤 User: \"{text}\"")
    time.sleep(0.3)


# ── Phase 1: Meal Planning (T1 Planner) ──────────────────────────

def phase1_plan_meals() -> list[str]:
    """User requests meals → T1 planner generates ingredient list."""
    banner("PHASE 1: What do I want to cook?")

    user_says("I want to make carbonara and a stir fry this week")
    say("Great choices! Let me plan those meals for you...")

    from sous_bot.planner.engine import PlannerEngine
    planner = PlannerEngine()

    # Use T1 planner to generate ingredients
    from sous_bot.api.schemas import ChatMessage
    history: list[ChatMessage] = []
    reply, plan, recipe = planner.chat_with_plan(
        history,
        "I want to make carbonara and chicken stir fry. "
        "List ALL ingredients I need for both recipes.",
        [],
    )

    say(reply)

    # Extract needed ingredients from the plan or use fallback
    if plan and plan.shopping_list:
        needed = [item.name for item in plan.shopping_list]
        say(f"You'll need {len(needed)} ingredients total across both recipes.")
    else:
        # Fallback: common ingredients for these recipes
        needed = [
            "pasta", "eggs", "guanciale", "parmesan", "black pepper",
            "olive oil", "chicken", "soy sauce", "ginger", "bell pepper",
            "rice", "garlic", "sesame oil",
        ]
        say(f"For carbonara and stir fry, you'll need about {len(needed)} ingredients.")

    return needed


# ── Phase 2: Pantry Scan (T3 Vision) ─────────────────────────────

def phase2_scan_pantry(
    needed: list[str], image_path: str | None = None,
) -> list[str]:
    """Scan pantry → detect ingredients → compute missing."""
    banner("PHASE 2: What do I already have?")

    from sous_bot.vision.detector import IngredientDetector
    from sous_bot.vision.inventory import InventoryTracker

    tracker = InventoryTracker()
    detected_names: list[str] = []

    if image_path:
        user_says("Here's my pantry, take a look")
        say("Scanning your pantry with the camera...")

        from sous_bot.vision.camera import CameraCapture
        image_bytes = CameraCapture.load_image(image_path)

        detector = IngredientDetector()
        result = detector.detect_from_bytes(image_bytes)

        if result.ingredients:
            detected_names = [i.name for i in result.ingredients]
            items_str = ", ".join(detected_names)
            say(f"I can see {len(detected_names)} items: {items_str}.")
        else:
            say("Hmm, I couldn't detect much. Let me use some defaults.")

    if not detected_names:
        # Demo fallback: simulate a pantry scan
        user_says("Let me show you my pantry")
        say("Scanning your pantry...")
        time.sleep(1)

        detected_names = ["pasta", "eggs", "parmesan", "rice", "soy sauce", "garlic"]
        items_str = ", ".join(detected_names)
        say(f"I can see {len(detected_names)} items: {items_str}.")

    # Compute missing
    tracker.update_available(detected_names)
    tracker.set_needed(needed)
    inv = tracker.get_inventory()

    if inv.missing:
        say(f"You're missing {len(inv.missing)} items: {', '.join(inv.missing)}.")
    else:
        say("You have everything you need!")

    return inv.missing


# ── Phase 3: Robot Shopping (T2 Simulated) ────────────────────────

def phase3_robot_shopping(missing: list[str]) -> None:
    """Simulated robot fetches items from grocery store."""
    banner("PHASE 3: Go shopping!")

    if not missing:
        say("No shopping needed — your pantry is fully stocked!")
        return

    from sous_bot.vision.inventory import InventoryTracker

    tracker = InventoryTracker()
    tracker.set_needed(missing)  # Only missing items need fetching

    user_says("Show me the shopping")
    say(f"Sending the robot to fetch {len(missing)} items. Let's go!")

    # Simulated store aisles
    aisle_map = {
        "guanciale": "deli", "chicken": "deli", "ground beef": "deli",
        "black pepper": "spices", "ginger": "spices", "garlic": "spices",
        "olive oil": "oils", "sesame oil": "oils",
        "bell pepper": "produce", "lettuce": "produce", "onion": "produce",
        "tortillas": "bakery", "cheese": "dairy", "butter": "dairy",
        "soy sauce": "condiments", "salsa": "condiments",
    }

    for i, item in enumerate(missing, 1):
        aisle = aisle_map.get(item, "aisle 3")

        # Step 3a: Navigate
        print(f"\n  📍 [{i}/{len(missing)}] Navigating to {aisle} section...")
        time.sleep(0.3)

        # Step 3b: Vision locates item
        print(f"  👁️  Scanning shelf... Found {item}!")
        time.sleep(0.2)

        # Step 3c: Reach and grasp
        print(f"  🦾 Reaching for {item}... Grasped!")
        time.sleep(0.2)

        # Step 3d: Place in cart
        result = tracker.add_to_cart(item)
        print(f"  🛒 Placed {item} in cart. ({len(result.collected)}/{len(missing)})")
        time.sleep(0.2)

        if not result.complete:
            remaining = ", ".join(result.remaining)
            print(f"     Still need: {remaining}")

    # Step 3e: Validate
    print()
    result = tracker.validate_cart()
    if result.complete:
        say(f"Shopping complete! Got all {len(result.collected)} items. "
            "All requirements met! Heading to checkout.")
    else:
        say(f"Almost done. Still missing: {', '.join(result.remaining)}.")


# ── Main Demo ─────────────────────────────────────────────────────

def main() -> None:
    print("""
╔══════════════════════════════════════════════════════╗
║                                                      ║
║           🤖  SOUS BOT — Hackathon Demo  🛒          ║
║                                                      ║
║   Assistive grocery robot for visually impaired      ║
║   and elderly users. Built at Nebius.Build SF.       ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
    """)

    say("Hello! I'm Sous Bot, your personal grocery assistant. "
        "Let me show you what I can do.", pause=1.0)

    # Get image path if provided
    image_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--image" and i + 1 < len(sys.argv):
            image_path = sys.argv[i + 1]

    # Phase 1: Plan meals
    needed = phase1_plan_meals()

    # Phase 2: Scan pantry
    missing = phase2_scan_pantry(needed, image_path)

    # Phase 3: Robot shopping
    phase3_robot_shopping(missing)

    # Wrap up
    banner("DEMO COMPLETE")
    say("That's the full Sous Bot flow! "
        "Voice in, vision scan, AI planning, and robot shopping. "
        "Thank you for watching!", pause=1.5)

    # Optional interactive mode
    if "--interactive" in sys.argv:
        print("\n--- Entering interactive mode (type 'quit' to exit) ---")
        from sous_bot.voice.assistant import VoiceAssistant
        assistant = VoiceAssistant(use_mic=False)
        # Pre-load the inventory state
        for item in (needed or []):
            assistant._inventory.add_available(item) if item not in (missing or []) else None
        if needed:
            assistant._inventory.set_needed(needed)
        assistant.run()


if __name__ == "__main__":
    main()
