"""Voice I/O pipeline composing STT and TTS."""

from __future__ import annotations

from pathlib import Path

from sous_bot.voice.stt import TranscriptionResult, WhisperSTT
from sous_bot.voice.tts import TextToSpeech


class VoicePipeline:
    """End-to-end voice input/output pipeline for accessibility."""

    def __init__(
        self,
        stt: WhisperSTT | None = None,
        tts: TextToSpeech | None = None,
    ) -> None:
        self._stt = stt or WhisperSTT()
        self._tts = tts or TextToSpeech()

    def process_audio(self, audio_bytes: bytes) -> TranscriptionResult:
        """Transcribe audio input to text."""
        return self._stt.transcribe_bytes(audio_bytes)

    def process_audio_file(self, path: str | Path) -> TranscriptionResult:
        """Transcribe an audio file to text."""
        return self._stt.transcribe_file(path)

    def respond(self, text: str) -> None:
        """Speak a text response aloud."""
        self._tts.speak(text)
