"""
Quick audio-quality diagnostic for input.wav.

Run AFTER you've done one voice exchange with the fixed server:

    python check_audio.py

What it checks:
  1. Basic WAV format (rate / channels / sample width) - should be
     16000 Hz, mono, 16-bit.
  2. Peak / RMS levels.
  3. A simple "even/odd sample correlation" test. If the old bug is
     present (interleaved L/R treated as mono), the even-indexed and
     odd-indexed samples come from two DIFFERENT physical channels,
     so they tend to look like uncorrelated/alternating noise. On
     genuinely fixed mono speech, consecutive samples are part of the
     same continuous waveform and correlate normally.
  4. Runs Whisper on the file and prints the transcript, so you can
     confirm end-to-end.
"""

import sys
import wave
import numpy as np
from pathlib import Path

WAV_PATH = Path(__file__).resolve().parent / "input.wav"


def load_wav(path):
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)

    if width != 2:
        raise ValueError(f"Expected 16-bit PCM, got sampwidth={width}")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    return audio, rate, channels


def main():
    if not WAV_PATH.exists():
        print(f"Could not find {WAV_PATH}")
        print("Run a voice exchange through the server first.")
        sys.exit(1)

    audio, rate, channels = load_wav(WAV_PATH)

    print("=" * 50)
    print("WAV FORMAT")
    print("=" * 50)
    print(f"Sample rate : {rate} Hz  (expected 16000)")
    print(f"Channels    : {channels}  (expected 1)")
    print(f"Duration    : {len(audio) / rate:.2f} sec")

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0

    print()
    print("=" * 50)
    print("LEVELS")
    print("=" * 50)
    print(f"Peak : {peak:.5f}")
    print(f"RMS  : {rms:.5f}")

    # --------------------------------------------------------
    # Even/odd interleaving check.
    #
    # If L/R channels got interleaved and treated as mono, the
    # even and odd sample streams are effectively two different
    # (and often anti-correlated, since mic channels can be near
    # -identical or phase-shifted) signals stitched together.
    # A short-lag autocorrelation gap between the "same channel"
    # lag (2 samples) and the "adjacent" lag (1 sample) tends to
    # be unusually large when interleaving corruption is present.
    # This is a heuristic, not a hard proof -- use it as a signal,
    # then trust your ears.
    # --------------------------------------------------------

    if len(audio) > 2000:
        lag1 = np.corrcoef(audio[:-1], audio[1:])[0, 1]
        lag2 = np.corrcoef(audio[:-2], audio[2:])[0, 1]

        print()
        print("=" * 50)
        print("INTERLEAVING HEURISTIC")
        print("=" * 50)
        print(f"Correlation at lag=1: {lag1:.3f}")
        print(f"Correlation at lag=2: {lag2:.3f}")

        if lag2 > lag1 + 0.15:
            print(
                "WARNING: lag-2 correlation is notably higher than "
                "lag-1. This pattern is consistent with interleaved "
                "stereo samples being treated as mono (the bug we "
                "fixed). If you still see this after updating "
                "server.py, double check you're running the new file."
            )
        else:
            print("No interleaving pattern detected. Looks like clean mono audio.")

    # --------------------------------------------------------
    # Whisper transcript (optional, only if whisper is installed)
    # --------------------------------------------------------

    print()
    print("=" * 50)
    print("WHISPER TRANSCRIPT")
    print("=" * 50)

    try:
        import whisper

        model = whisper.load_model("base")
        result = model.transcribe(str(WAV_PATH), language="en", fp16=False)
        text = result.get("text", "").strip()
        print(repr(text) if text else "(empty)")

    except Exception as e:
        print(f"Skipped (whisper not available or errored): {e!r}")


if __name__ == "__main__":
    main()