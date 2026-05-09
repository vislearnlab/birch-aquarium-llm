"""Microphone capture + local Whisper transcription."""
from __future__ import annotations

import os
import sys

# Anaconda's numpy and ctranslate2 (faster-whisper) ship separate OpenMP runtimes
# that collide on macOS. The duplicate-init guard is conservative and the
# duplication is benign for our single-threaded use; allow it.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from . import config

_MODEL: WhisperModel | None = None


def _model() -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        print(f"loading whisper model: {config.WHISPER_MODEL} (first run downloads ~150MB)")
        _MODEL = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    return _MODEL


def record_until_enter() -> np.ndarray:
    """Capture mic audio until the user presses Enter. Returns float32 mono @ 16kHz."""
    chunks: list[np.ndarray] = []

    def callback(indata, frames, time, status):
        if status:
            print(f"[audio status: {status}]", file=sys.stderr)
        chunks.append(indata.copy())

    stream = sd.InputStream(
        samplerate=config.AUDIO_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=callback,
    )
    with stream:
        try:
            input()  # block until Enter
        except (EOFError, KeyboardInterrupt):
            pass

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([c.flatten() for c in chunks]).astype(np.float32)


def transcribe(audio: np.ndarray) -> str:
    """Transcribe audio to text. Returns '' if input is too short."""
    min_samples = int(0.3 * config.AUDIO_SAMPLE_RATE)  # 300ms
    if audio.size < min_samples:
        return ""
    segments, _ = _model().transcribe(audio, language="en", vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


def listen() -> str:
    """One full record + transcribe cycle. Returns the transcribed text."""
    print("  [press Enter to start recording]", end="", flush=True)
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        return ""
    print("  [recording... press Enter to stop]", end="", flush=True)
    audio = record_until_enter()
    if audio.size == 0:
        return ""
    print("  [transcribing...]", end="\r", flush=True)
    return transcribe(audio)
