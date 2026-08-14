# session.py

import os
import json
import requests

from tts import speak


# ============================================================
# CONFIG
# ============================================================

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

MODEL = "openai/gpt-oss-20b"


# ============================================================
# CHECK API KEY
# ============================================================

if not NVIDIA_API_KEY:
    raise RuntimeError(
        "NVIDIA_API_KEY is not set.\n"
        "Run this in CMD:\n"
        "set NVIDIA_API_KEY=YOUR_KEY"
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a real-time voice assistant.

Rules:
- Speak naturally.
- Keep answers concise.
- Do not use markdown.
- Do not use bullet points unless absolutely necessary.
- Do not mention that you are an AI unless asked.
- Answer directly.
- Your responses will be converted to speech.
"""


# ============================================================
# SESSION
# ============================================================

class VoiceSession:

    def __init__(self):

        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

    # ========================================================
    # ASK GPT
    # ========================================================

    def ask(self, text: str) -> str:

        text = text.strip()

        if not text:
            return ""

        print("\n🤖 Asking NVIDIA GPT-OSS-20B...")
        print("Sending request to NVIDIA...")

        # ----------------------------------------------------
        # Add user message
        # ----------------------------------------------------

        self.messages.append(
            {
                "role": "user",
                "content": text,
            }
        )

        payload = {
            "model": MODEL,
            "messages": self.messages,
            "max_tokens": 300,
            "temperature": 0.4,
            "stream": True,
        }

        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        # ====================================================
        # SEND REQUEST
        # ====================================================

        try:

            response = requests.post(
                NVIDIA_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=(10, 120),
            )

            response.raise_for_status()

        except requests.exceptions.Timeout:

            print("\n❌ NVIDIA request timed out")

            return (
                "Sorry, the AI service took too long "
                "to respond."
            )

        except requests.exceptions.RequestException as e:

            print("\n❌ NVIDIA request failed:")
            print(repr(e))

            return (
                "Sorry, I couldn't connect "
                "to the AI service."
            )

        # ====================================================
        # READ STREAM
        # ====================================================

        answer = ""

        try:

            for line in response.iter_lines():

                if not line:
                    continue

                line = line.decode("utf-8")

                # NVIDIA SSE:
                # data: {...}

                if not line.startswith("data: "):
                    continue

                data = line[6:]

                if data == "[DONE]":
                    break

                try:

                    chunk = json.loads(data)

                except json.JSONDecodeError:

                    continue

                choices = chunk.get(
                    "choices",
                    []
                )

                if not choices:
                    continue

                delta = choices[0].get(
                    "delta",
                    {}
                )

                # ------------------------------------------------
                # GPT-OSS may send reasoning.
                #
                # We only collect "content".
                # Reasoning is NOT sent to TTS.
                # ------------------------------------------------

                content = delta.get("content")

                if content:

                    print(
                        content,
                        end="",
                        flush=True
                    )

                    answer += content

        except Exception as e:

            print(
                "\n❌ Error reading NVIDIA stream:"
            )

            print(repr(e))

            return (
                "Sorry, I had trouble generating "
                "a response."
            )

        print()

        # ====================================================
        # CLEAN RESPONSE
        # ====================================================

        answer = answer.strip()

        if not answer:

            answer = (
                "Sorry, I couldn't generate a response."
            )

        # ====================================================
        # SAVE ASSISTANT MESSAGE
        # ====================================================

        self.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return answer


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("VOICE SESSION TEST")
    print("=" * 50)

    session = VoiceSession()

    while True:

        try:
            text = input("\nYou: ")

        except KeyboardInterrupt:
            print("\nBye!")
            break

        if text.lower().strip() in [
            "exit",
            "quit",
            "bye",
        ]:
            print("Bye!")
            break

        if not text.strip():
            continue

        # GPT
        answer = session.ask(text)

        print("\nAssistant:", answer)

        # TTS
        print("\n🔊 Starting TTS...")

        try:
            speak(answer)
            print("✅ TTS finished")
            print("💾 Check speech.wav")

        except Exception as e:
            print("❌ TTS ERROR:")
            print(repr(e))