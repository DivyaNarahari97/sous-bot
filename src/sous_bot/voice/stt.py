"""Speech-to-text using OpenAI Whisper (local)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel, Field


class TranscriptionResult(BaseModel):
    """Result from speech-to-text transcription."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    language: str = "en"


class WhisperSTT:
    """Local Whisper speech-to-text engine with lazy loading."""

    def __init__(self, model_name: str = "base") -> None:
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazily load the Whisper model on first use."""
        if self._model is None:
            import whisper

            self._model = whisper.load_model(self._model_name)

    def transcribe_file(self, path: str | Path) -> TranscriptionResult:
        """Transcribe an audio file to text."""
        self._load_model()
        result = self._model.transcribe(str(path))
        return TranscriptionResult(
            text=result["text"].strip(),
            language=result.get("language", "en"),
        )

    def transcribe_bytes(self, audio_bytes: bytes) -> TranscriptionResult:
        """Transcribe raw audio bytes (WAV format expected)."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            return self.transcribe_file(tmp.name)
