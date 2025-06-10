# utils/audio_tools.py
"""
Audio tools for Buddy using Termux:API
"""
import subprocess
import os

AUDIO_PATH = "/data/data/com.termux/files/home/data/audio/command.wav"

class transcribe_audio:
    @staticmethod
    def record_to_file(duration=5, path=AUDIO_PATH):
        print("🎙️ Starting recording...")
        if os.path.exists(path):
            os.remove(path)

        subprocess.run([
            "termux-microphone-record",
            "-l", str(duration),
            "-f", path
        ])

        print("🎙️ Recording complete.")
        return path

    @staticmethod
    def transcribe(audio_path=AUDIO_PATH):
        # Placeholder: simulate transcription result
        return "start trip 48220. odometer 705000. trailer 123456"


def speak_text(text: str):
    print("🔊 Buddy says:", text)
    subprocess.run(["termux-tts-speak", text])
