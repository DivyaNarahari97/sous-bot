"""T3 Voice FastAPI routes — T1 imports these into the main app."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from sous_bot.voice.stt import TranscriptionResult, WhisperSTT
from sous_bot.voice.tts import TextToSpeech

router = APIRouter(prefix="/voice", tags=["voice"])

# Shared state — singleton instances
_stt: WhisperSTT | None = None
_tts: TextToSpeech | None = None


def _get_stt() -> WhisperSTT:
    global _stt
    if _stt is None:
        _stt = WhisperSTT()
    return _stt


def _get_tts() -> TextToSpeech:
    global _tts
    if _tts is None:
        _tts = TextToSpeech()
    return _tts


# ── Request/Response schemas ──────────────────────────────────────


class SpeakRequest(BaseModel):
    text: str


class SpeakResponse(BaseModel):
    status: str = "ok"
    text: str = ""


# ── Endpoints ─────────────────────────────────────────────────────


@router.post("/transcribe", response_model=TranscriptionResult)
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe uploaded audio file to text using Whisper STT."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty audio file")

    stt = _get_stt()
    return stt.transcribe_bytes(contents)


@router.post("/speak", response_model=SpeakResponse)
async def speak_text(request: SpeakRequest):
    """Speak text aloud via TTS (for accessibility)."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    tts = _get_tts()
    tts.speak(request.text)
    return SpeakResponse(text=request.text)
