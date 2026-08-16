from piper import PiperVoice
import wave
import os
import sys

MODEL = "en_US-lessac-medium.onnx"
OUTPUT = "speech.wav"

voice = PiperVoice.load(MODEL)


def speak(text):
    text = text.strip()

    if not text:
        print("No text to speak.")
        return

    print("Generating speech...")

    with wave.open(OUTPUT, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    print("Saved:", OUTPUT)

    # ------------------------------------------------------------
    # NOTE:
    #
    # We no longer play this audio locally on the server via
    # PowerShell/SoundPlayer. Playing it through the server
    # machine's speakers was being picked up by the browser's
    # microphone (feedback) and/or briefly disrupting the mic
    # input device, which was killing the WebRTC audio track
    # right after every response.
    #
    # server.py now streams this wav file back to the browser
    # over the WebRTC connection itself (see AssistantAudioTrack),
    # so it plays through the *browser's* <audio> element instead.
    # ------------------------------------------------------------


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = "Hello, this is your voice agent."
    speak(text)