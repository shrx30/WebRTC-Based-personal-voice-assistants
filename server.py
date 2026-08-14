import asyncio
import wave

import numpy as np
import torch
from scipy.signal import resample_poly

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError

from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, get_speech_timestamps

from session import VoiceSession
from tts import speak


# ============================================================
# CONFIG
# ============================================================

HOST = "127.0.0.1"
PORT = 8000

TARGET_SAMPLE_RATE = 16000

VAD_THRESHOLD = 0.25
MIN_SPEECH_MS = 200
MIN_SILENCE_MS = 500
SPEECH_PAD_MS = 250

peer_connections = set()


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI()

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
# LOAD WHISPER
# ============================================================

print("=" * 50)
print("Loading Whisper STT model...")
print("=" * 50)

whisper_model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8",
)

print("Whisper loaded.")


# ============================================================
# LOAD SILERO
# ============================================================

print()
print("Loading Silero VAD...")

vad_model = load_silero_vad()

print("Silero VAD loaded.")


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

    # --------------------------------------------------------
    # DEBUG ONLY ON FIRST FRAME
    # --------------------------------------------------------

    if pcm.ndim == 1:

        mono_pcm = pcm.astype(
            np.int16
        )

    elif pcm.ndim == 2:

        # Usually:
        #
        # (channels, samples)
        #
        # For your microphone:
        #
        # (2, samples)

        if pcm.shape[0] <= 8:

            mono_pcm = (
                pcm.astype(
                    np.float32
                )
                .mean(axis=0)
            )

        else:

            # Defensive case:
            #
            # (samples, channels)

            mono_pcm = (
                pcm.astype(
                    np.float32
                )
                .mean(axis=1)
            )

        mono_pcm = np.rint(
            mono_pcm
        ).astype(np.int16)

    else:

        raise RuntimeError(
            f"Unexpected audio shape: {pcm.shape}"
        )

    # --------------------------------------------------------
    # INT16 -> FLOAT32
    # --------------------------------------------------------

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
# PROCESS UTTERANCE
# ============================================================

async def process_utterance(
    audio,
    sample_rate,
):

    print()
    print("-" * 50)
    print("Processing utterance")
    print("-" * 50)

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

    tensor = torch.from_numpy(
        audio_16k
    ).float()

    try:

        timestamps = get_speech_timestamps(
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
    # WHISPER
    # ========================================================

    print()
    print("Transcribing...")

    try:

        segments, info = (
            whisper_model.transcribe(

                speech_tensor.numpy(),

                language="en",

                beam_size=5,

                temperature=0,

                condition_on_previous_text=False,

                vad_filter=False,

                no_speech_threshold=0.4,
            )
        )

        transcript = ""

        for segment in segments:

            transcript += segment.text

        transcript = transcript.strip()

    except Exception as e:

        print(
            "Whisper error:",
            repr(e),
        )

        return

    print(
        "Detected language:",
        info.language,
    )

    print(
        "Language probability:",
        info.language_probability,
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
    # GPT
    # ========================================================

    print()
    print(
        "Sending transcript to GPT..."
    )

    try:

        answer = voice_session.ask(
            transcript
        )

    except Exception as e:

        print(
            "GPT error:",
            repr(e),
        )

        return

    print()
    print(
        "Assistant:",
        answer,
    )

    # ========================================================
    # TTS
    # ========================================================

    print()
    print(
        "Generating speech..."
    )

    try:

        speak(answer)

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


# ============================================================
# RECEIVE WEBRTC AUDIO
# ============================================================

async def receive_audio(
    peer_connection,
    track,
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
            # SILERO VAD
            # =================================================

            try:

                timestamps = get_speech_timestamps(
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
                    # PROCESS
                    # =========================================

                    if len(utterance) > 0:

                        await process_utterance(
                            utterance,
                            sample_rate,
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