"""
Whisper transcription worker.

Deliberately kept minimal — this module gets imported inside a
SEPARATE OS PROCESS (via ProcessPoolExecutor), isolated from
server.py's aiortc/FastAPI/torch-VAD code. That isolation is the
whole point: Whisper (CTranslate2, native C++) and aiortc's audio
pipeline (PyAV/FFmpeg, also native C++) were crashing when forced
to coexist in one process. Separate processes have separate
memory, so they can no longer collide.

Do NOT import aiortc, fastapi, or anything WebRTC-related here.
"""

from faster_whisper import WhisperModel

_model = None


def init_worker():
    """
    Runs once, automatically, when the worker process starts
    (passed as ProcessPoolExecutor's `initializer`). Loads the
    model once per worker process, not once per call.
    """

    global _model

    _model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
    )


def transcribe_audio(audio_np):
    """
    Runs inside the worker process. Takes a plain numpy float32
    array (picklable) and returns plain picklable values only —
    never the raw faster-whisper `info` object, since that isn't
    guaranteed picklable across the process boundary.
    """

    segments, info = _model.transcribe(
        audio_np,
        language="en",
        beam_size=5,
        temperature=0,
        condition_on_previous_text=False,
        vad_filter=False,
        no_speech_threshold=0.4,
    )

    transcript = ""

    for segment in segments:
        transcript += segment.text

    transcript = transcript.strip()

    return (
        transcript,
        info.language,
        info.language_probability,
    )