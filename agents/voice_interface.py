import os
import subprocess
from utils.audio_tools import transcribe_audio, speak_text

class VoiceInterface:
    def __init__(self, wake_word="buddy"):  # no wake word detection here yet
        self.wake_word = wake_word

    def listen_for_wake_word(self):
        print("🎤 Buddy is ready. Press ENTER to start recording your command...")
        while True:
            input("▶️ Press Enter to record...")
            self.record_command()

    def record_command(self):
        print("🎙️ Recording with termux-microphone-record (5 sec)...")
        audio_path = transcribe_audio.record_to_file(duration=5)
        command_text = transcribe_audio.transcribe(audio_path)

        print("📥 Command received:", command_text)
        response = "Got it! Working on your request..."  # placeholder
        speak_text(response)
