import subprocess
import json
from pathlib import Path
import re
from agents.voice_interface import transcribe_audio_whisper
from agents.core_router import CoreRouter
from trip_manager import log_trip_update


LOG_FILE = Path.home() / "trip_log.json"


def log_trip(command_text):
    """Extract trip info and save it to trip_log.json"""
    trip_data = {}

    # Use regex to extract basic trip info
    trip_match = re.search(r"trip (\d+)", command_text)
    odo_match = re.search(r"odometer (\d+)", command_text)
    trailer_match = re.search(r"trailer (\d+)", command_text)
    stops_match = re.search(r"(\d+) stop", command_text)
    type_match = re.search(r"drop|live load|live unload", command_text)

    if trip_match:
        trip_data["trip_number"] = trip_match.group(1)
    if odo_match:
        trip_data["odometer"] = odo_match.group(1)
    if trailer_match:
        trip_data["trailer"] = trailer_match.group(1)
    if stops_match:
        trip_data["stops"] = stops_match.group(1)
    if type_match:
        trip_data["trip_type"] = type_match.group(0).title()

    # Load existing logs
    if LOG_FILE.exists():
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    # Add new trip
    logs.append(trip_data)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

    return trip_data


def speak(text):
    """Speak Buddy's response out loud using Termux TTS."""
    subprocess.run(["termux-tts-speak", text])


def main():
    print("🎤 Buddy is loaded. Press Enter to begin voice input...")

    # Create a basic CoreRouter instance (adjust as needed)
    router = CoreRouter(llm_runner=None, voice_interface=None)

    while True:
        input("🔘 Press Enter when you're ready to speak...")

        text = transcribe_audio_whisper()

        if not text:
            print("❌ No voice input detected. Switching to text mode...")
            try:
                text = input("⌨️ Type your command: ")
            except KeyboardInterrupt:
                print("\n👋 Exiting Buddy.")
                break

        # Smarter trigger for trip logging
        if any(kw in text for kw in ["trip", "odometer", "drop trailer", "hook trailer", "from dc", "stop at", "final stop"]):
            response = log_trip_update(text)
        else:
            response = router.route_command(text)

        print(f"🧠 Buddy: {response}")
        speak(response)

        # Optional: voice-initiated exit
        if response.lower().startswith("🛑 stopping"):
            print("👋 Exiting Buddy by voice command.")
            break



if __name__ == "__main__":
    main()




