"""Voice module: STT (Whisper) and TTS (pyttsx3) for accessibility."""

from sous_bot.voice.assistant import VoiceAssistant
from sous_bot.voice.recorder import MicRecorder
from sous_bot.voice.stt import TranscriptionResult, WhisperSTT
from sous_bot.voice.tts import TextToSpeech

__all__ = [
    "MicRecorder",
    "TextToSpeech",
    "TranscriptionResult",
    "VoiceAssistant",
    "WhisperSTT",
]
