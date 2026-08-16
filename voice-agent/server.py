import asyncio
import fractions
import time
import wave
import os
import faulthandler
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from scipy.signal import resample_poly

from av import AudioFrame

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack

from silero_vad import load_silero_vad, get_speech_timestamps

import whisper_worker

from session import VoiceSession
from tts import speak
from fastapi import FastAPI
from fastapi.responses import FileResponse

# ============================================================
# CONFIG
# ============================================================

HOST = "127.0.0.1"
PORT = 8000

# Dumps a low-level C-stack trace to crash_log.txt if the process
# dies from a native fault (segfault/access violation) — the kind
# of crash a normal try/except can never catch. Purely diagnostic.
_fault_log = open("crash_log.txt", "w")
faulthandler.enable(file=_fault_log, all_threads=True)

TARGET_SAMPLE_RATE = 16000

VAD_THRESHOLD = 0.25
MIN_SPEECH_MS = 200
MIN_SILENCE_MS = 500
SPEECH_PAD_MS = 250

# Hard ceiling on how long a single utterance's VAD buffer can
# grow before we force-finalize it (safety net against runaway
# per-frame VAD cost during a long ramble).
MAX_UTTERANCE_SECONDS = 20

# Outgoing (assistant -> browser) audio track settings.
OUTPUT_SAMPLE_RATE = 48000
OUTPUT_FRAME_MS = 20
OUTPUT_SAMPLES_PER_FRAME = int(
    OUTPUT_SAMPLE_RATE * OUTPUT_FRAME_MS / 1000
)  # 960 samples per 20ms frame at 48kHz

peer_connections = set()

# Prevents a second utterance from being processed (Whisper/GPT/TTS)
# while a previous one is still running.
processing_lock = asyncio.Lock()

# Piper (ONNX) and the GPT/NVIDIA network call are pinned to ONE
# dedicated worker thread rather than asyncio.to_thread's default
# rotating pool. Costs nothing performance-wise since
# processing_lock already fully serializes these calls anyway.
blocking_executor = ThreadPoolExecutor(max_workers=1)


async def run_blocking(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(blocking_executor, func, *args)


# Whisper runs in a fully separate OS PROCESS (whisper_worker.py),
# not a thread — it was crashing (silent, native, no traceback)
# when sharing a process with aiortc's audio pipeline. Deliberately
# NOT created here at module level: on Windows, a ProcessPoolExecutor
# spawned this way gets RE-CREATED when the child process re-imports
# this script, causing recursive process spawning. It's created once,
# lazily, inside warm_up_models() below, which only ever runs in the
# real main process (after uvicorn.run() has actually started).
whisper_executor = None


async def run_whisper(audio_np):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        whisper_executor,
        whisper_worker.transcribe_audio,
        audio_np,
    )


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI()
@app.get("/")
async def serve_index():
    return FileResponse(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "index.html"
        )
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# OFFER MODEL
# ============================================================

class Offer(BaseModel):
    sdp: str
    type: str


# ============================================================
# WHISPER
# ============================================================
#
# Whisper now runs in a completely separate OS PROCESS (see
# whisper_worker.py), not loaded here. This is deliberate: it
# was crashing (silent, native, no traceback) when the model
# lived in this same process as aiortc's audio pipeline. See
# whisper_executor / warm_up_models() below for where the
# worker process actually gets started.

print("Whisper will run in a separate worker process (see whisper_worker.py).")


# ============================================================
# LOAD SILERO
# ============================================================

print()
print("Loading Silero VAD...")

vad_model = load_silero_vad()

print("Silero VAD loaded.")

# vad_model is one shared object called from two independent
# places: receive_audio()'s continuous per-frame loop, and
# process_utterance()'s own VAD pass on the finalized utterance.
# Both dispatch to real OS threads via asyncio.to_thread, and
# PyTorch's CPU backend is NOT safe to reenter concurrently from
# two threads on the same model — doing so corrupted the process
# heap (confirmed via crash_log.txt: two threads simultaneously
# inside torch calls at the moment of a STATUS_HEAP_CORRUPTION
# crash). This lock makes sure only one call into vad_model can
# be in flight at any moment, process-wide.
vad_lock = asyncio.Lock()


# ============================================================
# VOICE SESSION
# ============================================================

voice_session = VoiceSession()


# ============================================================
# RESAMPLE
# ============================================================

def resample_audio(
    audio,
    original_rate,
):
    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if original_rate == TARGET_SAMPLE_RATE:
        return audio

    return resample_poly(
        audio,
        TARGET_SAMPLE_RATE,
        original_rate,
    ).astype(np.float32)


# ============================================================
# SAVE MONO FLOAT WAV
# ============================================================

def save_wav(
    filename,
    audio,
    sample_rate,
):
    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    audio = np.clip(
        audio,
        -1.0,
        1.0,
    )

    pcm = (
        audio * 32767.0
    ).astype(np.int16)

    with wave.open(
        filename,
        "wb",
    ) as wav:

        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        wav.writeframes(
            pcm.tobytes()
        )


# ============================================================
# CONVERT WEBRTC FRAME TO MONO FLOAT32
# ============================================================

def frame_to_mono(frame):

    # IMPORTANT:
    #
    # Do NOT use:
    #
    # frame.to_ndarray(format="s16")
    #
    # Your PyAV version doesn't support that argument.
    #
    # The WebRTC frame is already reported as:
    #
    # format = s16
    #
    # so simply use to_ndarray().

    pcm = frame.to_ndarray()

    channels = len(frame.layout.channels)
    samples = frame.samples

    if channels == 1:

        # Mono track — nothing to de-interleave.

        mono_pcm = pcm.reshape(-1).astype(
            np.int16
        )

    elif (
        pcm.ndim == 2
        and pcm.shape[0] == 1
        and pcm.shape[1] == samples * channels
    ):

        # PACKED / INTERLEAVED case:
        #
        # A single row containing
        # L R L R L R ... samples back to back.
        #
        # This is what your browser mic sends:
        # e.g. shape (1, 1920) for
        # 960 samples x 2 channels.
        #
        # Reshape to (samples, channels) BEFORE
        # averaging, otherwise left/right samples
        # get treated as sequential mono samples,
        # which sounds muddy/garbled and breaks
        # transcription.

        stereo = pcm.reshape(
            -1
        ).reshape(
            samples,
            channels,
        )

        mono_pcm = (
            stereo.astype(
                np.float32
            )
            .mean(axis=1)
        )

        mono_pcm = np.rint(
            mono_pcm
        ).astype(np.int16)

    elif (
        pcm.ndim == 2
        and pcm.shape[0] == channels
    ):

        # PLANAR case:
        #
        # (channels, samples) — each channel
        # already in its own row.

        mono_pcm = (
            pcm.astype(
                np.float32
            )
            .mean(axis=0)
        )

        mono_pcm = np.rint(
            mono_pcm
        ).astype(np.int16)

    else:

        raise RuntimeError(
            f"Unexpected audio shape: {pcm.shape}, "
            f"channels={channels}, samples={samples}"
        )

    audio = (
        mono_pcm.astype(
            np.float32
        )
        / 32768.0
    )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    audio = np.clip(
        audio,
        -1.0,
        1.0,
    )

    return audio


# ============================================================
# ASSISTANT (OUTGOING) AUDIO TRACK
# ============================================================
#
# This is what actually sends the assistant's voice to the
# browser over the WebRTC connection. It's a persistent audio
# track (added to the peer connection once, up front) that
# emits silence between responses and streams a wav file's
# audio, frame by frame, whenever push_wav() is called.
#
# It must keep returning frames at a steady 20ms cadence for
# the whole life of the connection — aiortc calls recv() in a
# loop and paces sending based on when each call returns.
# ============================================================

class AssistantAudioTrack(MediaStreamTrack):

    kind = "audio"

    # How long after the last real (non-silence) frame we still
    # treat the assistant as "speaking." Covers echo/reverb that
    # lingers briefly after playback actually stops, so the mic
    # doesn't immediately start reacting to the tail of its own
    # voice.
    SPEAKING_GRACE_SECONDS = 0.6

    def __init__(self):

        super().__init__()

        self._queue = asyncio.Queue()

        self._samples_sent = 0
        self._start_time = None

        self._last_real_chunk_time = None

    def is_speaking(self):

        if not self._queue.empty():
            return True

        if self._last_real_chunk_time is None:
            return False

        return (
            time.time() - self._last_real_chunk_time
        ) < self.SPEAKING_GRACE_SECONDS

    async def push_wav(self, path):

        # Load whatever sample rate/channel count Piper wrote,
        # convert to mono int16 at OUTPUT_SAMPLE_RATE, split
        # into fixed-size frames, and enqueue them.

        with wave.open(path, "rb") as wf:

            channels = wf.getnchannels()
            rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())

        pcm = np.frombuffer(
            raw,
            dtype=np.int16,
        )

        if channels > 1:

            pcm = (
                pcm.reshape(-1, channels)
                .astype(np.float32)
                .mean(axis=1)
            )

        else:

            pcm = pcm.astype(np.float32)

        if rate != OUTPUT_SAMPLE_RATE:

            pcm = resample_poly(
                pcm,
                OUTPUT_SAMPLE_RATE,
                rate,
            )

        pcm = np.clip(
            pcm,
            -32768,
            32767,
        ).astype(np.int16)

        # Pad so it splits evenly into fixed-size frames.

        remainder = len(pcm) % OUTPUT_SAMPLES_PER_FRAME

        if remainder:

            pad = OUTPUT_SAMPLES_PER_FRAME - remainder

            pcm = np.concatenate(
                [pcm, np.zeros(pad, dtype=np.int16)]
            )

        for i in range(
            0,
            len(pcm),
            OUTPUT_SAMPLES_PER_FRAME,
        ):

            chunk = pcm[
                i : i + OUTPUT_SAMPLES_PER_FRAME
            ]

            await self._queue.put(chunk)

    async def recv(self):

        if self._start_time is None:
            self._start_time = time.time()

        try:

            chunk = self._queue.get_nowait()

            self._last_real_chunk_time = time.time()

        except asyncio.QueueEmpty:

            # Nothing to say right now — send silence so the
            # track (and therefore the connection) stays alive.

            chunk = np.zeros(
                OUTPUT_SAMPLES_PER_FRAME,
                dtype=np.int16,
            )

        frame = AudioFrame(
            format="s16",
            layout="mono",
            samples=OUTPUT_SAMPLES_PER_FRAME,
        )

        frame.planes[0].update(
            chunk.tobytes()
        )

        frame.sample_rate = OUTPUT_SAMPLE_RATE

        frame.pts = self._samples_sent

        frame.time_base = fractions.Fraction(
            1,
            OUTPUT_SAMPLE_RATE,
        )

        self._samples_sent += (
            OUTPUT_SAMPLES_PER_FRAME
        )

        # Pace ourselves to real time so frames go out roughly
        # every 20ms instead of as fast as the loop can spin.

        target_time = (
            self._start_time
            + self._samples_sent / OUTPUT_SAMPLE_RATE
        )

        delay = target_time - time.time()

        if delay > 0:
            await asyncio.sleep(delay)

        return frame


# ============================================================
# BLOCKING WORK, RUN VIA asyncio.to_thread()
# ============================================================
#
# GPT/NVIDIA and TTS synthesis are synchronous, network/CPU-bound
# calls. If they ran directly on the asyncio event loop, they'd
# freeze aiortc's packet handling for as long as they take,
# starving the incoming audio track. run_blocking() moves them
# onto a dedicated worker thread so the event loop stays free.
#
# Whisper is handled separately — see run_whisper() above, which
# delegates to a fully separate OS process (whisper_worker.py).
# ============================================================

def run_gpt(transcript):
    return voice_session.ask(transcript)


def run_tts(answer):
    speak(answer)


# ============================================================
# WARM UP MODELS AT STARTUP
# ============================================================
#
# Whisper (CTranslate2) and Piper (ONNX) both do real work the
# first time they're called — kernel/graph initialization — on
# top of just running inference. Doing that warm-up here, once,
# at server startup, means your FIRST real utterance isn't the
# one that eats the delay.

@app.on_event("startup")
async def warm_up_models():

    global whisper_executor

    print()
    print(
        "Starting Whisper worker process..."
    )

    whisper_executor = ProcessPoolExecutor(
        max_workers=1,
        initializer=whisper_worker.init_worker,
    )

    print(
        "Warming up Whisper and TTS "
        "(first call is normally slow)..."
    )

    try:

        dummy_audio = np.zeros(
            16000,
            dtype=np.float32,
        )

        await run_whisper(
            dummy_audio,
        )

        print("Whisper warmed up.")

    except Exception as e:

        print(
            "Whisper warm-up failed (non-fatal):",
            repr(e),
        )

    try:

        await run_blocking(
            run_tts,
            "Warming up.",
        )

        print("TTS warmed up.")

    except Exception as e:

        print(
            "TTS warm-up failed (non-fatal):",
            repr(e),
        )

    print(
        "Warm-up complete — ready for connections."
    )
    print()


# ============================================================
# PROCESS UTTERANCE
# ============================================================

async def process_utterance(
    audio,
    sample_rate,
    assistant_track,
):

    async with processing_lock:

        print()
        print("-" * 50)
        print("Processing utterance")
        print("-" * 50)

        stage_times = {}
        t_utterance_start = time.time()

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        duration = (
            len(audio)
            / sample_rate
        )

        peak = float(
            np.max(
                np.abs(audio)
            )
        )

        rms = float(
            np.sqrt(
                np.mean(
                    audio ** 2
                )
            )
        )

        print(
            f"Raw audio duration: "
            f"{duration:.2f} seconds"
        )

        print(
            "Raw peak:",
            peak,
        )

        print(
            "Raw RMS:",
            rms,
        )

        # ========================================================
        # SAVE RAW WEBRTC AUDIO
        # ========================================================

        try:

            save_wav(
                "webrtc_raw.wav",
                audio,
                sample_rate,
            )

            print(
                "Saved: webrtc_raw.wav"
            )

        except Exception as e:

            print(
                "Could not save raw WAV:",
                repr(e),
            )

        # ========================================================
        # RESAMPLE TO 16K
        # ========================================================

        print(
            "Resampling audio..."
        )

        audio_16k = resample_audio(
            audio,
            sample_rate,
        )

        print(
            "Audio prepared:",
            audio_16k.shape,
        )

        # ========================================================
        # SAVE 16K FULL AUDIO
        # ========================================================

        try:

            save_wav(
                "input_full.wav",
                audio_16k,
                16000,
            )

            print(
                "Saved: input_full.wav"
            )

        except Exception as e:

            print(
                "Could not save input_full.wav:",
                repr(e),
            )

        # ========================================================
        # VAD
        # ========================================================

        print(
            "Detecting speech..."
        )

        t0 = time.time()

        tensor = torch.from_numpy(
            audio_16k
        ).float()

        try:

            async with vad_lock:

                timestamps = await asyncio.to_thread(
                    get_speech_timestamps,
                    tensor,
                    vad_model,
                    sampling_rate=16000,
                    threshold=VAD_THRESHOLD,
                    min_speech_duration_ms=MIN_SPEECH_MS,
                    min_silence_duration_ms=MIN_SILENCE_MS,
                    speech_pad_ms=SPEECH_PAD_MS,
                )

        except Exception as e:

            print(
                "VAD error:",
                repr(e),
            )

            return

        stage_times["VAD"] = time.time() - t0

        print(
            "Speech segments:",
            len(timestamps),
        )

        if not timestamps:

            print(
                "No speech detected."
            )

            return

        # ========================================================
        # EXTRACT SPEECH
        # ========================================================

        speech_parts = []

        for ts in timestamps:

            start = ts["start"]
            end = ts["end"]

            speech_parts.append(
                tensor[start:end]
            )

        if not speech_parts:

            print(
                "No speech audio."
            )

            return

        speech_tensor = torch.cat(
            speech_parts
        )

        speech_duration = (
            len(speech_tensor)
            / 16000
        )

        print(
            f"Speech duration: "
            f"{speech_duration:.2f} seconds"
        )

        print(
            "Speech samples:",
            speech_tensor.shape,
        )

        # ========================================================
        # SAVE WHISPER INPUT
        # ========================================================

        try:

            save_wav(
                "input.wav",
                speech_tensor.numpy(),
                16000,
            )

            print(
                "Saved: input.wav"
            )

        except Exception as e:

            print(
                "Could not save input.wav:",
                repr(e),
            )

            return

        # ========================================================
        # WHISPER (runs in its own separate OS process — isolated
        # from aiortc's audio pipeline)
        # ========================================================

        print()
        print("Transcribing...")

        t0 = time.time()

        try:

            transcript, language, language_probability = await run_whisper(
                speech_tensor.numpy(),
            )

        except Exception as e:

            print(
                "Whisper error:",
                repr(e),
            )

            return

        stage_times["Whisper"] = time.time() - t0

        print(
            "Detected language:",
            language,
        )

        print(
            "Language probability:",
            language_probability,
        )

        print()
        print(
            "Transcript:",
            transcript,
        )

        if not transcript:

            print(
                "Empty transcript."
            )

            return

        # ========================================================
        # GPT (runs in a worker thread — does not block aiortc)
        # ========================================================

        print()
        print(
            "Sending transcript to GPT..."
        )

        t0 = time.time()

        try:

            answer = await run_blocking(
                run_gpt,
                transcript,
            )

        except Exception as e:

            print(
                "GPT error:",
                repr(e),
            )

            return

        stage_times["GPT"] = time.time() - t0

        print()
        print(
            "Assistant:",
            answer,
        )

        # ========================================================
        # TTS (runs in a worker thread — does not block aiortc)
        # ========================================================

        print()
        print(
            "Generating speech..."
        )

        t0 = time.time()

        try:

            await run_blocking(
                run_tts,
                answer,
            )

            stage_times["TTS"] = time.time() - t0

            print(
                "TTS finished"
            )

            print(
                "Check speech.wav"
            )

        except Exception as e:

            print(
                "TTS error:",
                repr(e),
            )

            return

        # ========================================================
        # SEND ASSISTANT AUDIO BACK OVER WEBRTC
        # ========================================================

        try:

            await assistant_track.push_wav(
                "speech.wav"
            )

            print(
                "Queued speech.wav for playback to browser"
            )

        except Exception as e:

            print(
                "Failed to queue assistant audio:",
                repr(e),
            )

        # ========================================================
        # TIMING BREAKDOWN
        # ========================================================

        total = time.time() - t_utterance_start

        print()
        print(
            "⏱  Latency breakdown "
            "(silence-to-speaking-again):"
        )

        for stage_name, elapsed in stage_times.items():

            print(
                f"    {stage_name:8s} "
                f"{elapsed:6.2f}s"
            )

        print(
            f"    {'TOTAL':8s} "
            f"{total:6.2f}s"
        )
        print()


# ============================================================
# RECEIVE WEBRTC AUDIO
# ============================================================

async def receive_audio(
    peer_connection,
    track,
    assistant_track,
):

    print()
    print(
        "Listening continuously..."
    )

    audio_buffer = []
    vad_buffer = []

    sample_rate = None

    speech_active = False

    frame_count = 0

    while True:

        try:

            frame = await track.recv()

            frame_count += 1

            # =================================================
            # FIRST FRAME
            # =================================================

            if sample_rate is None:

                sample_rate = (
                    frame.sample_rate
                )

                print()
                print(
                    "Audio sample rate:",
                    sample_rate,
                )

                print(
                    "Audio format:",
                    frame.format.name,
                )

                print(
                    "Channels:",
                    len(
                        frame.layout.channels
                    ),
                )

                print(
                    "Frame samples:",
                    frame.samples,
                )

                print(
                    "Frame shape:",
                    frame.to_ndarray().shape,
                )

            # =================================================
            # WEBRTC -> MONO FLOAT32
            # =================================================

            audio = frame_to_mono(
                frame
            )

            # =================================================
            # IGNORE MIC WHILE ASSISTANT IS TALKING
            # =================================================
            #
            # Without this, the browser mic picks up the
            # assistant's own reply coming out of the speakers
            # (echo/feedback) and VAD gets confused right after
            # every response — detecting spurious "speech,"
            # missing real silence, or producing garbage/empty
            # transcripts.

            if assistant_track.is_speaking():

                if (
                    audio_buffer
                    or vad_buffer
                    or speech_active
                ):

                    audio_buffer = []
                    vad_buffer = []

                    speech_active = False

                continue

            # =================================================
            # BUFFER
            # =================================================

            audio_buffer.append(
                audio
            )

            vad_buffer.append(
                audio
            )

            # =================================================
            # VAD BUFFER
            # =================================================

            vad_audio = np.concatenate(
                vad_buffer
            )

            # Need enough audio before checking VAD

            if len(vad_audio) < int(
                sample_rate * 0.25
            ):

                continue

            # =================================================
            # SAFETY CAP
            # =================================================
            #
            # get_speech_timestamps() re-scans the WHOLE
            # buffer from scratch on every single frame.
            # Without a ceiling, a long ramble makes each call
            # progressively more expensive, which (combined
            # with it running directly on the event loop)
            # starves both this track and the outgoing
            # AssistantAudioTrack's 20ms pacing — looks like
            # the app has "frozen." Force-finalize instead of
            # letting the buffer grow without bound.

            if len(vad_audio) > int(
                sample_rate * MAX_UTTERANCE_SECONDS
            ):

                print(
                    "\n⚠️ Max utterance length reached — "
                    "forcing finalize"
                )

                complete_audio = np.concatenate(
                    audio_buffer
                )

                utterance = complete_audio

                audio_buffer = []
                vad_buffer = []

                speech_active = False

                if len(utterance) > 0:

                    asyncio.create_task(
                        process_utterance(
                            utterance,
                            sample_rate,
                            assistant_track,
                        )
                    )

                print()
                print(
                    "Listening continuously..."
                )

                continue

            # =================================================
            # RESAMPLE VAD AUDIO
            # =================================================

            vad_16k = resample_audio(
                vad_audio,
                sample_rate,
            )

            vad_tensor = (
                torch.from_numpy(
                    vad_16k
                ).float()
            )

            # =================================================
            # SILERO VAD (off the event loop — this is CPU
            # bound and must not block track.recv() / the
            # outgoing audio track's pacing. vad_lock ensures
            # this never overlaps with process_utterance's own
            # VAD call on the same shared vad_model — running
            # both at once corrupted the process heap.)
            # =================================================

            try:

                async with vad_lock:

                    timestamps = await asyncio.to_thread(
                        get_speech_timestamps,
                        vad_tensor,
                        vad_model,
                        sampling_rate=16000,
                        threshold=VAD_THRESHOLD,
                        min_speech_duration_ms=MIN_SPEECH_MS,
                        min_silence_duration_ms=MIN_SILENCE_MS,
                        speech_pad_ms=SPEECH_PAD_MS,
                    )

            except Exception as e:

                print(
                    "VAD error:",
                    repr(e),
                )

                continue

            # =================================================
            # SPEECH
            # =================================================

            if timestamps:

                if not speech_active:

                    print(
                        "\n🎤 Speech detected"
                    )

                    speech_active = True

            # =================================================
            # SPEECH END
            # =================================================

            if (
                speech_active
                and timestamps
            ):

                last = timestamps[-1]

                speech_end = last["end"]

                total_samples = (
                    len(vad_tensor)
                )

                silence_samples = (
                    total_samples
                    - speech_end
                )

                silence_seconds = (
                    silence_samples
                    / 16000.0
                )

                if (
                    silence_seconds
                    >= 0.7
                ):

                    print(
                        "\n🛑 Speech ended"
                    )

                    # =========================================
                    # COMBINE ORIGINAL-RATE AUDIO
                    # =========================================

                    complete_audio = (
                        np.concatenate(
                            audio_buffer
                        )
                    )

                    # =========================================
                    # TIMESTAMPS ARE 16K
                    # CONVERT TO ORIGINAL RATE
                    # =========================================

                    start_original = int(
                        timestamps[0]["start"]
                        * sample_rate
                        / 16000
                    )

                    end_original = int(
                        timestamps[-1]["end"]
                        * sample_rate
                        / 16000
                    )

                    # =========================================
                    # PADDING
                    # =========================================

                    padding = int(
                        0.25
                        * sample_rate
                    )

                    start_original = max(
                        0,
                        start_original
                        - padding,
                    )

                    end_original = min(
                        len(complete_audio),
                        end_original
                        + padding,
                    )

                    utterance = (
                        complete_audio[
                            start_original:end_original
                        ]
                    )

                    # =========================================
                    # RESET
                    # =========================================

                    audio_buffer = []
                    vad_buffer = []

                    speech_active = False

                    # =========================================
                    # PROCESS — fire-and-forget task.
                    #
                    # This is the key lifecycle fix: we do NOT
                    # `await` process_utterance here. Awaiting
                    # it would block this loop (and therefore
                    # block track.recv()) for as long as
                    # Whisper/GPT/TTS take to run, starving
                    # aiortc and killing the track. Scheduling
                    # it as a background task lets this loop
                    # go straight back to consuming frames,
                    # while process_utterance's own blocking
                    # calls run off-loop via asyncio.to_thread.
                    # =========================================

                    if len(utterance) > 0:

                        asyncio.create_task(
                            process_utterance(
                                utterance,
                                sample_rate,
                                assistant_track,
                            )
                        )

                    print()
                    print(
                        "Listening continuously..."
                    )

        except MediaStreamError:

            print(
                "\n🎤 Microphone track ended"
            )

            break

        except asyncio.CancelledError:

            break

        except Exception as e:

            print(
                "\n❌ Audio error:",
                repr(e),
            )

            break


# ============================================================
# WEBRTC OFFER
# ============================================================

@app.post("/offer")
async def offer(
    data: Offer,
):

    print()
    print(
        "=" * 50
    )

    print(
        "WEBRTC OFFER RECEIVED"
    )

    print(
        "=" * 50
    )

    pc = RTCPeerConnection()

    peer_connections.add(
        pc
    )

    print(
        "PeerConnection created"
    )

    # ========================================================
    # OUTGOING (ASSISTANT) AUDIO TRACK
    # ========================================================
    #
    # Must be added BEFORE createAnswer() so the SDP answer
    # actually negotiates a sendable audio m-line matching the
    # browser's recvonly transceiver.

    assistant_track = AssistantAudioTrack()

    pc.addTrack(assistant_track)

    print(
        "Assistant audio track added"
    )

    # ========================================================
    # TRACK
    # ========================================================

    @pc.on("track")
    def on_track(track):

        if track.kind == "audio":

            print(
                "\n🎤 Audio track received"
            )

            asyncio.create_task(
                receive_audio(
                    pc,
                    track,
                    assistant_track,
                )
            )

    # ========================================================
    # CONNECTION
    # ========================================================

    @pc.on(
        "connectionstatechange"
    )
    async def on_connectionstatechange():

        print(
            "Connection:",
            pc.connectionState,
        )

        if pc.connectionState in {
            "failed",
            "closed",
        }:

            try:

                await pc.close()

            except Exception:

                pass

            peer_connections.discard(
                pc
            )

    # ========================================================
    # ICE
    # ========================================================

    @pc.on(
        "iceconnectionstatechange"
    )
    async def on_iceconnectionstatechange():

        print(
            "ICE:",
            pc.iceConnectionState,
        )

    # ========================================================
    # REMOTE DESCRIPTION
    # ========================================================

    await pc.setRemoteDescription(
        RTCSessionDescription(
            sdp=data.sdp,
            type=data.type,
        )
    )

    print(
        "Remote description set"
    )

    # ========================================================
    # ANSWER
    # ========================================================

    answer = await pc.createAnswer()

    await pc.setLocalDescription(
        answer
    )

    print(
        "Answer created"
    )

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    }


# ============================================================
# SHUTDOWN
# ============================================================

@app.on_event("shutdown")
async def shutdown():

    print(
        "\nShutting down..."
    )

    for pc in list(
        peer_connections
    ):

        try:

            await pc.close()

        except Exception:

            pass

    peer_connections.clear()

    if whisper_executor is not None:

        whisper_executor.shutdown(
            wait=False,
            cancel_futures=True,
        )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    print()
    print(
        "=" * 50
    )

    print(
        "VOICE AGENT WEBRTC SERVER"
    )

    print(
        "=" * 50
    )

    print(
        f"Server: http://{HOST}:{PORT}"
    )

    print(
        f"Offer: http://{HOST}:{PORT}/offer"
    )

    print(
        "=" * 50
    )

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
    )