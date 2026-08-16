import asyncio
import fractions
import requests
import numpy as np
import sounddevice as sd

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc import MediaStreamTrack
from av import AudioFrame


SERVER_URL = "http://127.0.0.1:8000/offer"


# ============================================================
# MICROPHONE
# ============================================================

class MicrophoneTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()

        self.sample_rate = 48000
        self.channels = 1
        self.block_size = 960

        self.queue = asyncio.Queue(maxsize=30)
        self.loop = asyncio.get_running_loop()

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.block_size,
            callback=self._callback,
        )

        self.stream.start()

        print("🎤 Microphone started")

    def _callback(self, indata, frames, time, status):

        if status:
            print("Microphone:", status)

        data = indata.copy()

        try:
            self.loop.call_soon_threadsafe(
                self._put_audio,
                data
            )
        except Exception:
            pass

    def _put_audio(self, data):

        try:
            self.queue.put_nowait(data)
        except asyncio.QueueFull:
            pass

    async def recv(self):

        data = await self.queue.get()

        # sounddevice gives:
        # (samples, channels)

        mono = data[:, 0]

        # aiortc/PyAV expects:
        # (channels, samples)

        array = mono.reshape(1, -1)

        frame = AudioFrame.from_ndarray(
            array,
            format="s16",
            layout="mono",
        )

        frame.sample_rate = self.sample_rate

        frame.pts = 0
        frame.time_base = fractions.Fraction(
            1,
            self.sample_rate
        )

        return frame

    def stop(self):

        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass

        super().stop()


# ============================================================
# PLAY SERVER TTS
# ============================================================

async def receive_audio(track):

    print("🔊 TTS audio track received")

    try:

        while True:

            frame = await track.recv()

            audio = frame.to_ndarray()

            # PyAV normally gives:
            #
            # mono:
            # (1, samples)
            #
            # stereo:
            # (channels, samples)

            if audio.ndim == 2:

                audio = audio.T

            else:

                audio = audio.reshape(-1, 1)

            # Convert to int16

            if audio.dtype != np.int16:
                audio = audio.astype(np.int16)

            sample_rate = frame.sample_rate or 22050

            channels = audio.shape[1]

            print(
                f"🔊 Audio: "
                f"{sample_rate} Hz, "
                f"{channels} channel(s), "
                f"{len(audio)} samples"
            )

            sd.play(
                audio,
                samplerate=sample_rate,
                blocking=True
            )

    except asyncio.CancelledError:
        pass

    except Exception as e:
        print(
            "❌ TTS playback error:",
            repr(e)
        )


# ============================================================
# MAIN
# ============================================================

async def main():

    print("=" * 60)
    print("VOICE AGENT WEBRTC CLIENT")
    print("=" * 60)

    print(
        "Server:",
        SERVER_URL
    )

    print()

    pc = RTCPeerConnection()

    microphone = MicrophoneTrack()

    # ========================================================
    # MICROPHONE -> SERVER
    # ========================================================

    pc.addTrack(microphone)

    print(
        "🎤 Microphone track attached."
    )

    # ========================================================
    # SERVER -> CLIENT AUDIO
    # ========================================================

    pc.addTransceiver(
        "audio",
        direction="recvonly"
    )

    # ========================================================
    # RECEIVE AUDIO TRACK
    # ========================================================

    @pc.on("track")
    def on_track(track):

        print(
            f"📡 Received remote track: {track.kind}"
        )

        if track.kind == "audio":

            asyncio.create_task(
                receive_audio(track)
            )

    # ========================================================
    # CONNECTION EVENTS
    # ========================================================

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():

        print(
            "Connection:",
            pc.connectionState
        )

        if pc.connectionState == "connected":

            print()
            print("🟢 WEBRTC CONNECTED")
            print("🎤 Speak now.")
            print("Press CTRL+C to stop.")
            print()

        elif pc.connectionState == "failed":

            print(
                "❌ WebRTC connection failed."
            )

    @pc.on("iceconnectionstatechange")
    def on_iceconnectionstatechange():

        print(
            "ICE:",
            pc.iceConnectionState
        )

    @pc.on("icegatheringstatechange")
    def on_icegatheringstatechange():

        print(
            "ICE gathering:",
            pc.iceGatheringState
        )

    # ========================================================
    # CREATE OFFER
    # ========================================================

    print(
        "Creating WebRTC offer..."
    )

    offer = await pc.createOffer()

    await pc.setLocalDescription(
        offer
    )

    # ========================================================
    # WAIT FOR ICE
    # ========================================================

    print(
        "Waiting for ICE gathering..."
    )

    while pc.iceGatheringState != "complete":

        await asyncio.sleep(0.1)

    print(
        "ICE gathering complete."
    )

    # ========================================================
    # SEND OFFER TO SERVER
    # ========================================================

    print(
        "Sending offer to server..."
    )

    payload = {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    }

    try:

        response = requests.post(
            SERVER_URL,
            json=payload,
            timeout=30,
        )

    except Exception as e:

        print(
            "❌ Could not connect to server:"
        )

        print(
            repr(e)
        )

        microphone.stop()
        await pc.close()

        return

    if response.status_code != 200:

        print(
            "❌ Server returned:",
            response.status_code
        )

        print(
            response.text
        )

        microphone.stop()
        await pc.close()

        return

    answer = response.json()

    # ========================================================
    # SET SERVER ANSWER
    # ========================================================

    await pc.setRemoteDescription(
        RTCSessionDescription(
            sdp=answer["sdp"],
            type=answer["type"],
        )
    )

    print(
        "✅ Remote description set."
    )

    print()
    print(
        "Listening..."
    )
    print()

    # ========================================================
    # KEEP CONNECTION ALIVE
    # ========================================================

    try:

        while True:

            await asyncio.sleep(1)

    except asyncio.CancelledError:
        pass

    except KeyboardInterrupt:
        pass

    finally:

        print()
        print(
            "Stopping client..."
        )

        microphone.stop()

        await pc.close()

        sd.stop()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "\nClient stopped."
        )

    except Exception as e:

        print()
        print(
            "❌ Client error:"
        )
        print(
            repr(e)
        )