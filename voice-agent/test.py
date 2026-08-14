import asyncio
import av
import numpy as np
import soundfile as sf

MIC_NAME = "Headset (OnePlus Nord Buds 3 Pro)"


async def main():
    print(f"Starting: {MIC_NAME}")

    container = av.open(
        f"audio={MIC_NAME}",
        format="dshow",
        mode="r",
    )

    print("Microphone opened.")
    print("Recording 5 seconds...")
    print("SPEAK NOW!")

    samples = []
    sample_rate = None

    start = asyncio.get_running_loop().time()

    try:
        while asyncio.get_running_loop().time() - start < 5:

            frame = next(container.decode(audio=0))

            print(
                f"Frame: {frame.format.name} "
                f"shape={frame.to_ndarray().shape} "
                f"dtype={frame.to_ndarray().dtype}"
            )

            audio = frame.to_ndarray()

            # Convert to mono
            if audio.ndim == 2:
                audio = audio[0]

            # Convert to float32
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            else:
                audio = audio.astype(np.float32)

            samples.append(audio)

            if frame.sample_rate:
                sample_rate = frame.sample_rate

    finally:
        container.close()

    if not samples:
        print("❌ No audio captured.")
        return

    audio = np.concatenate(samples)

    print()
    print("================================")
    print("Recording complete")
    print("================================")
    print("Sample rate:", sample_rate)
    print("Samples:", len(audio))
    print("Duration:", len(audio) / sample_rate)
    print("Peak:", np.max(np.abs(audio)))
    print("RMS:", np.sqrt(np.mean(audio ** 2)))

    sf.write(
        "mic_test.wav",
        audio,
        sample_rate,
        subtype="PCM_16",
    )

    print()
    print("✅ Saved: mic_test.wav")
    print("Listen to mic_test.wav")


if __name__ == "__main__":
    asyncio.run(main())