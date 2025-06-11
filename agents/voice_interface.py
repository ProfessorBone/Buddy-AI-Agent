import subprocess
import os
from pathlib import Path

PROJECT_DIR = Path.home() / "whisper.cpp"
RAW_AUDIO = PROJECT_DIR / "raw_input.wav"
CLEAN_AUDIO = PROJECT_DIR / "clean.wav"
TEXT_OUTPUT = PROJECT_DIR / "clean.wav.txt"
WHISPER_CLI = Path.home() / "whisper.cpp/build/bin/whisper-cli"
MODEL_PATH = PROJECT_DIR / "models/ggml-tiny.en.bin"

def transcribe_audio_whisper():
    # Step 1: Record
    print("🎤 Recording voice...")
    subprocess.run([
        "termux-microphone-record",
        "-f", str(RAW_AUDIO),
        "-l", "12",
        "-e", "aac"
    ], check=True)
    
    # Step 2: Convert to PCM WAV
    print("🔁 Converting audio...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(RAW_AUDIO),
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(CLEAN_AUDIO)
    ], check=True)
    
    # Step 3: Transcribe with Whisper
    print("🧠 Transcribing...")
    subprocess.run([
        str(WHISPER_CLI),
        "-m", str(MODEL_PATH),
        "-f", str(CLEAN_AUDIO),
        "-l", "en",
        "-otxt"
    ], check=True)

    # Step 4: Return result
    if TEXT_OUTPUT.exists():
        with open(TEXT_OUTPUT, "r") as f:
            text = f.read().strip()
        print(f"✅ Transcription: {text}")
        return text
    else:
        print("❌ Transcription failed: no text file created")
        return None

