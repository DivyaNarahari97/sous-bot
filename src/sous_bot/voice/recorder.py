"""Microphone recording for voice input."""

from __future__ import annotations

import io
import wave

import numpy as np
import sounddevice as sd


class MicRecorder:
    """Records audio from the system microphone."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "int16",
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._dtype = dtype

    def record(self, duration: float = 5.0) -> bytes:
        """Record audio for a fixed duration, return WAV bytes."""
        print(f"Recording for {duration}s...")
        audio = sd.rec(
            int(duration * self._sample_rate),
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype=self._dtype,
        )
        sd.wait()
        print("Recording complete.")
        return self._to_wav(audio)

    def record_until_silence(
        self,
        silence_threshold: float = 500.0,
        silence_duration: float = 1.5,
        max_duration: float = 30.0,
        chunk_duration: float = 0.5,
    ) -> bytes:
        """Record until silence is detected, return WAV bytes."""
        print("Listening... (speak now)")
        chunks: list[np.ndarray] = []
        silent_time = 0.0
        total_time = 0.0
        chunk_samples = int(chunk_duration * self._sample_rate)

        while total_time < max_duration:
            chunk = sd.rec(
                chunk_samples,
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype=self._dtype,
            )
            sd.wait()
            chunks.append(chunk)
            total_time += chunk_duration

            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
            if rms < silence_threshold:
                silent_time += chunk_duration
            else:
                silent_time = 0.0

            # Only stop on silence after we've captured some speech
            if silent_time >= silence_duration and total_time > silence_duration + 0.5:
                break

        if not chunks:
            return self._to_wav(np.zeros((0, self._channels), dtype=self._dtype))

        audio = np.concatenate(chunks, axis=0)
        print(f"Captured {total_time:.1f}s of audio.")
        return self._to_wav(audio)

    def _to_wav(self, audio: np.ndarray) -> bytes:
        """Convert numpy audio array to WAV bytes."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self._channels)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(self._sample_rate)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()
