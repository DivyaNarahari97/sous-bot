"""Voice assistant loop integrating vision, planner (T1), and robotics (T2).

This module lives in T3's scope and provides the voice-driven user interface.
It defines callback protocols so T1 (planner) and T2 (robotics) can plug in
their implementations without T3 needing to modify their code.

Three modes of planner integration:
  1. Direct protocol callback (in-process)
  2. HTTP client calling T1's FastAPI endpoints (--api-url flag)
  3. Fallback Nebius LLM (standalone, no T1 needed)
"""

from __future__ import annotations

import os
from typing import Protocol

import requests as http_requests
from dotenv import load_dotenv
from openai import OpenAI

from sous_bot.vision.camera import CameraCapture
from sous_bot.vision.detector import IngredientDetector
from sous_bot.vision.inventory import CartValidation, InventoryTracker
from sous_bot.voice.recorder import MicRecorder
from sous_bot.voice.stt import WhisperSTT
from sous_bot.voice.tts import TextToSpeech

load_dotenv()

# ── Integration protocols (T1 & T2 implement these) ──────────────────


class PlannerCallback(Protocol):
    """Protocol for T1 planner integration."""

    def plan_meal(self, available: list[str]) -> str:
        """Given available ingredients, return a meal suggestion."""
        ...

    def get_shopping_list(self, available: list[str]) -> list[dict]:
        """Return list of ShoppingItem dicts with name, quantity, aisle."""
        ...

    def chat(self, message: str, context: dict) -> str:
        """Handle a general conversational message."""
        ...


class RoboticsCallback(Protocol):
    """Protocol for T2 robotics integration."""

    def execute_shopping(self, items: list[dict]) -> str:
        """Trigger robot to fetch items; return status text."""
        ...


# ── T1 API client (calls T1's FastAPI over HTTP) ─────────────────


class T1ApiPlanner:
    """Calls T1's planner API endpoints over HTTP."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self._base_url = base_url.rstrip("/")
        self._session_id = "voice-session"

    def plan_meal(self, available: list[str]) -> str:
        resp = http_requests.post(
            f"{self._base_url}/chat",
            json={
                "session_id": self._session_id,
                "message": "Suggest a meal I can make with what I have.",
                "available_ingredients": available,
            },
            timeout=30,
        )
        if resp.ok:
            return resp.json().get("message", "No response from planner.")
        return "Planner API unavailable."

    def get_shopping_list(self, available: list[str]) -> list[dict]:
        # First ask planner to generate a plan
        resp = http_requests.get(
            f"{self._base_url}/shopping-list",
            params={"session_id": self._session_id},
            timeout=15,
        )
        if resp.ok:
            return resp.json().get("items", [])
        return []

    def chat(self, message: str, context: dict) -> str:
        resp = http_requests.post(
            f"{self._base_url}/chat",
            json={
                "session_id": self._session_id,
                "message": message,
                "available_ingredients": context.get("available", []),
            },
            timeout=30,
        )
        if resp.ok:
            return resp.json().get("message", "No response.")
        return "Planner API unavailable."


# ── Built-in LLM fallback (uses Nebius directly until T1 is ready) ───

ASSISTANT_PROMPT = (
    "You are Sous Bot, a helpful grocery assistant for visually impaired "
    "and elderly users. You help with meal planning, ingredient tracking, and "
    "grocery shopping. Keep responses SHORT and conversational (2-3 sentences). "
    "Speak naturally as if talking to someone."
)


class _FallbackPlanner:
    """Minimal Nebius LLM planner used until T1 builds the real one."""

    def __init__(self) -> None:
        self._client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=os.environ["NEBIUS_API_KEY"],
        )
        self._model = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    def plan_meal(self, available: list[str]) -> str:
        return self._ask(
            f"I have these ingredients: {', '.join(available)}. "
            "Suggest ONE simple meal I can make. Be brief."
        )

    def get_shopping_list(self, available: list[str]) -> list[dict]:
        # Fallback returns text, not structured ShoppingItems
        return []

    def chat(self, message: str, context: dict) -> str:
        ctx_parts = []
        if context.get("available"):
            ctx_parts.append(
                f"Available ingredients: {', '.join(context['available'])}"
            )
        if context.get("missing"):
            ctx_parts.append(
                f"Missing ingredients: {', '.join(context['missing'])}"
            )
        ctx_str = "\n".join(ctx_parts)
        full_msg = f"{ctx_str}\n\nUser says: {message}" if ctx_str else message
        return self._ask(full_msg)

    def _ask(self, user_msg: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": ASSISTANT_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=256,
            temperature=0.7,
        )
        return resp.choices[0].message.content or "Sorry, I couldn't process that."


class _FallbackRobotics:
    """Stub robotics until T2 builds the real controller."""

    def execute_shopping(self, items: list[dict]) -> str:
        names = [i["name"] if isinstance(i, dict) else str(i) for i in items]
        return (
            f"Robot demo not yet connected. Items to fetch: {', '.join(names)}. "
            "The robotics module will handle this once T2 is ready."
        )


# ── Voice Assistant ───────────────────────────────────────────────────


class VoiceAssistant:
    """Main voice command loop — the full Phase 1→2→3 orchestrator.

    Usage:
        # Standalone (fallback planner)
        assistant = VoiceAssistant()
        assistant.run()

        # With T1 API running on localhost:8000
        assistant = VoiceAssistant(planner=T1ApiPlanner("http://localhost:8000"))
        assistant.run()

        # With T1/T2 in-process
        assistant = VoiceAssistant(planner=my_planner, robotics=my_robot)
        assistant.run()
    """

    def __init__(
        self,
        planner: PlannerCallback | None = None,
        robotics: RoboticsCallback | None = None,
        use_mic: bool = True,
        whisper_model: str = "base",
    ) -> None:
        self._tts = TextToSpeech()
        self._stt = WhisperSTT(model_name=whisper_model)
        self._recorder = MicRecorder() if use_mic else None
        self._detector = IngredientDetector()
        self._camera = CameraCapture()
        self._inventory = InventoryTracker()
        self._planner = planner or _FallbackPlanner()
        self._robotics = robotics or _FallbackRobotics()
        self._use_mic = use_mic
        self._shopping_items: list[dict] = []  # ShoppingItem dicts from T1

    def run(self) -> None:
        """Start the interactive voice command loop."""
        self._say(
            "Hello! I'm Sous Bot, your grocery assistant. "
            "I can scan your pantry, plan meals, make a shopping list, "
            "and guide the robot. Say help for commands, or just talk to me."
        )

        while True:
            text = self._listen()
            if not text:
                continue

            cmd = text.lower().strip()
            print(f"\n> You said: {text}")

            if any(w in cmd for w in ["quit", "exit", "stop", "goodbye"]):
                self._say("Goodbye! Happy cooking!")
                break
            elif any(w in cmd for w in ["scan", "pantry", "camera", "what do i have"]):
                self._handle_scan()
            elif cmd.startswith(("plan", "suggest a meal")) or cmd in ("meal", "recipe"):
                self._handle_plan()
            elif any(w in cmd for w in ["shopping list", "need", "missing", "buy list", "what do i need"]):
                self._handle_shopping()
            elif any(w in cmd for w in ["robot", "fetch", "go shop", "show me the shopping"]):
                self._handle_robot()
            elif "cart" in cmd or "validate" in cmd or "check cart" in cmd:
                self._handle_cart_check()
            elif any(w in cmd for w in ["help", "what can you do"]):
                self._say(
                    "Here's what I can do. "
                    "Say 'scan pantry' to see what you have. "
                    "Say 'plan a meal' for recipe suggestions. "
                    "Say 'what do I need' for your shopping list. "
                    "Say 'go shopping' to send the robot. "
                    "Say 'check cart' to see what's been collected. "
                    "Or just ask me anything!"
                )
            else:
                self._handle_chat(text)

    # ── Phase 2: Pantry scanning ──────────────────────────────────

    def _handle_scan(self) -> None:
        """Scan pantry using camera + vision LLM."""
        self._say("Let me scan your pantry...")
        try:
            image_bytes = self._camera.capture_frame()
        except RuntimeError:
            self._say("Camera not available. Do you have a pantry image file?")
            return

        result = self._detector.detect_from_bytes(image_bytes)
        if not result.ingredients:
            self._say(
                "I couldn't identify any ingredients. "
                "Try pointing the camera at your pantry shelves."
            )
            return

        names = [i.name for i in result.ingredients]
        self._inventory.update_available(names)
        items_str = ", ".join(names)
        self._say(f"I can see {len(names)} items: {items_str}.")

    def scan_image_file(self, path: str) -> None:
        """Scan a static image file (for demo/testing without camera)."""
        self._say("Scanning image...")
        image_bytes = CameraCapture.load_image(path)
        result = self._detector.detect_from_bytes(image_bytes)
        if not result.ingredients:
            self._say("I couldn't identify any ingredients in that image.")
            return

        names = [i.name for i in result.ingredients]
        self._inventory.update_available(names)
        items_str = ", ".join(names)
        self._say(f"I can see: {items_str}.")

    # ── Phase 1: Meal planning ────────────────────────────────────

    def _handle_plan(self) -> None:
        """Ask planner for a meal suggestion based on available ingredients."""
        inv = self._inventory.get_inventory()
        if not inv.available:
            self._say(
                "I haven't scanned your pantry yet. "
                "Say 'scan pantry' first, or I'll suggest something general."
            )

        response = self._planner.plan_meal(inv.available or ["nothing specific"])
        self._say(response)

    # ── Phase 2→3 bridge: Shopping list ───────────────────────────

    def _handle_shopping(self) -> None:
        """Get missing items / shopping list with aisle info."""
        inv = self._inventory.get_inventory()

        # Try to get structured shopping list from T1
        self._shopping_items = self._planner.get_shopping_list(inv.available or [])

        if self._shopping_items:
            # T1 returned structured ShoppingItem list — read with aisles
            count = len(self._shopping_items)
            self._say(f"You need {count} items.")

            # Set needed items in inventory for cart tracking
            needed_names = [item["name"] for item in self._shopping_items]
            self._inventory.set_needed(
                needed_names + list(inv.available)
            )

            # Read each item with aisle info
            for item in self._shopping_items:
                name = item.get("name", "unknown")
                qty = item.get("quantity", "")
                aisle = item.get("aisle", "")
                if aisle:
                    self._say(f"{qty} {name}, from the {aisle} section.")
                else:
                    self._say(f"{qty} {name}.")
        elif inv.missing:
            # Fallback: use inventory missing list (no quantities/aisles)
            items_str = ", ".join(inv.missing)
            self._say(f"You're missing {len(inv.missing)} items: {items_str}.")
        else:
            # Ask planner for general suggestions
            response = self._planner.chat(
                "What basic items should I buy for a good dinner?",
                {"available": inv.available},
            )
            self._say(response)

    # ── Phase 3: Robot shopping ───────────────────────────────────

    def _handle_robot(self) -> None:
        """Trigger T2 robotics to fetch shopping list items."""
        inv = self._inventory.get_inventory()
        shopping = self._inventory.get_shopping_list()

        if not shopping and not self._shopping_items:
            self._say(
                "No shopping list yet. "
                "Say 'plan a meal' first, then 'what do I need'."
            )
            return

        # Use structured items if available, else just names
        items = self._shopping_items if self._shopping_items else [
            {"name": name, "quantity": "", "aisle": None} for name in shopping
        ]

        count = len(items)
        self._say(f"Sending the robot to fetch {count} items. Let's go shopping!")
        self._inventory.reset_cart()

        response = self._robotics.execute_shopping(items)
        self._say(response)

    def _handle_cart_check(self) -> None:
        """Check cart validation status — called during/after robot shopping."""
        result: CartValidation = self._inventory.validate_cart()

        if result.complete:
            self._say(
                f"Shopping complete! Got all {len(result.collected)} items. "
                "All requirements met!"
            )
        elif not result.collected and not result.remaining:
            self._say("The cart is empty. Start shopping first.")
        else:
            self._say(result.message)

    # ── General chat ──────────────────────────────────────────────

    def _handle_chat(self, message: str) -> None:
        """General conversation via planner LLM."""
        inv = self._inventory.get_inventory()
        context = {
            "available": inv.available,
            "missing": inv.missing,
        }
        response = self._planner.chat(message, context)
        self._say(response)

    # ── I/O helpers ───────────────────────────────────────────────

    def _listen(self) -> str | None:
        """Get user input via mic or text fallback."""
        if self._use_mic and self._recorder:
            try:
                audio = self._recorder.record_until_silence()
                result = self._stt.transcribe_bytes(audio)
                return result.text if result.text.strip() else None
            except Exception as e:
                print(f"Mic error: {e}. Falling back to text input.")
                return input("\nType your command: ").strip() or None
        else:
            return input("\nType your command: ").strip() or None

    def _say(self, text: str) -> None:
        """Speak and print a response."""
        print(f"\nSous Bot: {text}")
        self._tts.speak(text)


def main() -> None:
    """Entry point for the voice assistant."""
    import sys

    use_mic = "--text" not in sys.argv
    whisper_model = "base"
    api_url = None

    for i, arg in enumerate(sys.argv):
        if arg == "--model" and i + 1 < len(sys.argv):
            whisper_model = sys.argv[i + 1]
        elif arg == "--api-url" and i + 1 < len(sys.argv):
            api_url = sys.argv[i + 1]

    # Choose planner: T1 API if available, else fallback
    planner = T1ApiPlanner(api_url) if api_url else None

    assistant = VoiceAssistant(
        planner=planner, use_mic=use_mic, whisper_model=whisper_model
    )

    # If an image path is provided, scan it first
    for i, arg in enumerate(sys.argv):
        if arg == "--image" and i + 1 < len(sys.argv):
            assistant.scan_image_file(sys.argv[i + 1])

    assistant.run()


if __name__ == "__main__":
    main()
