from piper import PiperVoice
import wave
import os
import sys
import subprocess

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

    # Automatically play the WAV on Windows
    print("Playing speech...")

    subprocess.run(
        ["powershell", "-c",
         f'(New-Object Media.SoundPlayer "{os.path.abspath(OUTPUT)}").PlaySync()'],
        check=False
    )

    print("Playback finished.")


if __name__ == "__main__":

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = "Hello, this is your voice agent."

    speak(text)