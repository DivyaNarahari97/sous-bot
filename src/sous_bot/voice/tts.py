"""Text-to-speech using pyttsx3."""

from __future__ import annotations

from pathlib import Path

import pyttsx3


class TextToSpeech:
    """System TTS engine for accessible voice output."""

    def __init__(self, rate: int = 150) -> None:
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)

    def speak(self, text: str) -> None:
        """Speak text aloud (blocking)."""
        self._engine.say(text)
        self._engine.runAndWait()

    def save_to_file(self, text: str, path: str | Path) -> None:
        """Save spoken text to an audio file."""
        self._engine.save_to_file(text, str(path))
        self._engine.runAndWait()
