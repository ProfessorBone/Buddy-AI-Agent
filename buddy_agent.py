import subprocess
from agents.voice_interface import transcribe_audio_whisper
from agents.core_router import CoreRouter


def speak(text):
    """Speak Buddy's response out loud using Termux TTS."""
    subprocess.run(["termux-tts-speak", text])


def main():
    print("🎤 Buddy is listening. Say your command...")

    # Create a basic CoreRouter instance (adjust these if needed)
    router = CoreRouter(llm_runner=None, voice_interface=None)

    while True:
        text = transcribe_audio_whisper()

        if not text:
            print("❌ No voice input detected. Switching to text mode...")
            try:
                text = input("⌨️ Type your command: ")
            except KeyboardInterrupt:
                print("\n👋 Exiting Buddy.")
                break

        response = router.route_command(text)
        print(f"🧠 Buddy: {response}")
        speak(response)


if __name__ == "__main__":
    main()


