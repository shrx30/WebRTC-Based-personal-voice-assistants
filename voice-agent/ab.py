"""
Isolation test: does Whisper crash on a SECOND transcribe() call
with no WebRTC / aiortc / AssistantAudioTrack involved at all?

Run this directly:

    python test_whisper_crash.py

If it crashes silently (no traceback, drops to prompt) on the
second call just like the server does, the bug is in Whisper /
CTranslate2 itself (or how it's being invoked), NOT in any
interaction with aiortc/audio-track code.

If it completes both calls fine, the bug is specifically in how
Whisper coexists with the rest of the server (most likely the
AssistantAudioTrack / aiortc audio pipeline running in the same
process), and we should look there instead.
"""

import sys
import numpy as np
import faulthandler

faulthandler.enable(all_threads=True)

from faster_whisper import WhisperModel


def load_wav_as_float32(path):
    import wave

    with wave.open(path, "rb") as wf:
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    pcm = np.frombuffer(raw, dtype=np.int16)
    audio = pcm.astype(np.float32) / 32768.0

    return audio, rate


def run_transcribe(model, audio, label):
    print(f"\n--- Transcribing ({label}) ---")

    segments, info = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        temperature=0,
        condition_on_previous_text=False,
        vad_filter=False,
        no_speech_threshold=0.4,
    )

    text = "".join(seg.text for seg in segments).strip()

    print(f"[{label}] Detected language:", info.language)
    print(f"[{label}] Transcript:", text)


def main():
    # Point this at any existing wav from your project folder —
    # input.wav / input_full.wav / webrtc_raw.wav all work.
    # Doesn't matter if they're the same file used twice.
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "input.wav"

    print("Loading Whisper model (small, cpu, int8)...")

    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
    )

    print("Model loaded.")

    audio, rate = load_wav_as_float32(wav_path)

    print(f"Loaded {wav_path}: {len(audio)} samples @ {rate}Hz")

    # First call — expected to work based on server logs so far.
    run_transcribe(model, audio, "call #1")

    print("\nCall #1 completed successfully.")
    print("Now attempting call #2 (this is where the server crashes)...")

    # Second call — this is the one that's been crashing the
    # server every time, with no traceback at all.
    run_transcribe(model, audio, "call #2")

    print("\nCall #2 completed successfully too.")
    print("If you're reading this, BOTH calls succeeded — ")
    print("meaning Whisper itself is fine in isolation, and the")
    print("crash is specific to the full server / aiortc pipeline.")


if __name__ == "__main__":
    main()