"""Voice assistant loop integrating vision, planner (T1), and robotics (T2).

This module lives in T3's scope and provides the voice-driven user interface.
It defines callback protocols so T1 (planner) and T2 (robotics) can plug in
their implementations without T3 needing to modify their code.
"""

from __future__ import annotations

import os
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI

from sous_bot.vision.camera import CameraCapture
from sous_bot.vision.detector import IngredientDetector
from sous_bot.vision.inventory import InventoryTracker
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

    def get_shopping_list(self, available: list[str]) -> str:
        """Given available ingredients, return missing items text."""
        ...

    def chat(self, message: str, context: dict) -> str:
        """Handle a general conversational message."""
        ...


class RoboticsCallback(Protocol):
    """Protocol for T2 robotics integration."""

    def execute_shopping(self, items: list[str]) -> str:
        """Trigger robot to fetch items; return status text."""
        ...


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

    def get_shopping_list(self, available: list[str]) -> str:
        return self._ask(
            f"I have: {', '.join(available)}. "
            "What basic items am I probably missing for a good dinner? "
            "List 3-5 items briefly."
        )

    def chat(self, message: str, context: dict) -> str:
        ctx_parts = []
        if context.get("available"):
            ctx_parts.append(f"Available ingredients: {', '.join(context['available'])}")
        if context.get("missing"):
            ctx_parts.append(f"Missing ingredients: {', '.join(context['missing'])}")
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

    def execute_shopping(self, items: list[str]) -> str:
        return (
            f"Robot demo not yet connected. Items to fetch: {', '.join(items)}. "
            "The robotics module will handle this once T2 is ready."
        )


# ── Voice Assistant ───────────────────────────────────────────────────


class VoiceAssistant:
    """Main voice command loop tying together vision, voice, and planner.

    Usage:
        assistant = VoiceAssistant()
        assistant.run()  # interactive mic loop

        # Or with T1/T2 plugged in:
        assistant = VoiceAssistant(planner=my_planner, robotics=my_robot)
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

    def run(self) -> None:
        """Start the interactive voice command loop."""
        self._say("Hello! I'm Sous Bot. You can say: scan pantry, "
                  "plan a meal, what do I need, or just ask me anything. "
                  "Say quit to exit.")

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
            elif any(w in cmd for w in ["shopping list", "need", "missing", "buy list"]):
                self._handle_shopping()
            elif any(w in cmd for w in ["robot", "fetch", "get items"]):
                self._handle_robot()
            elif any(w in cmd for w in ["help", "what can you do"]):
                self._say("I can scan your pantry, plan meals, tell you "
                          "what to buy, and show the robot shopping. "
                          "Just tell me what you'd like!")
            else:
                # Everything else goes to the LLM for natural conversation
                self._handle_chat(text)

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
            self._say("I couldn't identify any ingredients. "
                      "Try pointing the camera at your pantry shelves.")
            return

        names = [i.name for i in result.ingredients]
        self._inventory.update_available(names)
        items_str = ", ".join(names)
        self._say(f"I can see: {items_str}. "
                  f"That's {len(names)} items detected.")

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

    def _handle_plan(self) -> None:
        """Ask planner for a meal suggestion."""
        inv = self._inventory.get_inventory()
        if not inv.available:
            self._say("I haven't scanned your pantry yet. "
                      "Say 'scan pantry' first, or I'll suggest something general.")

        response = self._planner.plan_meal(inv.available or ["nothing specific"])
        self._say(response)

    def _handle_shopping(self) -> None:
        """Get missing items / shopping list."""
        inv = self._inventory.get_inventory()
        if inv.missing:
            items_str = ", ".join(inv.missing)
            self._say(f"You're missing: {items_str}.")
        else:
            response = self._planner.get_shopping_list(inv.available or [])
            self._say(response)

    def _handle_robot(self) -> None:
        """Trigger robotics to fetch items."""
        inv = self._inventory.get_inventory()
        items = inv.missing if inv.missing else ["some groceries"]
        response = self._robotics.execute_shopping(items)
        self._say(response)

    def _handle_chat(self, message: str) -> None:
        """General conversation via planner LLM."""
        inv = self._inventory.get_inventory()
        context = {
            "available": inv.available,
            "missing": inv.missing,
        }
        response = self._planner.chat(message, context)
        self._say(response)

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

    # Check for --model arg
    for i, arg in enumerate(sys.argv):
        if arg == "--model" and i + 1 < len(sys.argv):
            whisper_model = sys.argv[i + 1]

    assistant = VoiceAssistant(use_mic=use_mic, whisper_model=whisper_model)

    # If an image path is provided, scan it first
    for i, arg in enumerate(sys.argv):
        if arg == "--image" and i + 1 < len(sys.argv):
            assistant.scan_image_file(sys.argv[i + 1])

    assistant.run()


if __name__ == "__main__":
    main()
