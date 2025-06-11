import json
from pathlib import Path
import re

LOG_FILE = Path.home() / "trip_log.json"

def load_log():
    if LOG_FILE.exists():
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return []

def save_log(log):
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

def find_trip(log, trip_number):
    for trip in log:
        if trip.get("trip_number") == trip_number:
            return trip
    return None

def log_trip_update(command_text):
    command_text = command_text.lower()
    log = load_log()

    # Extract trip number
    trip_match = re.search(r"trip (\d+)", command_text)
    if not trip_match:
        return "❌ No trip number found."

    trip_number = trip_match.group(1)
    trip = find_trip(log, trip_number)

    if not trip:
        trip = {"trip_number": trip_number, "stops": [], "events": []}
        log.append(trip)

    # Add origin
    origin_match = re.search(r"from (dc \d+)", command_text)
    if origin_match:
        trip["origin"] = origin_match.group(1).upper()

    # Add stops
    stops = re.findall(r"store \d+|dc \d+|stop at (store \d+|dc \d+)|final stop at (store \d+|dc \d+)", command_text)
    for s in stops:
        loc = next(filter(None, s)) if isinstance(s, tuple) else s
        if loc and loc.upper() not in trip["stops"]:
            trip["stops"].append(loc.upper())

    # Drop event
    drop_match = re.search(r"drop trailer (\d+)(?: at ([\w\s\d]+))?", command_text)
    if drop_match:
        event = {"type": "drop", "trailer": drop_match.group(1)}
        if drop_match.group(2):
            event["location"] = drop_match.group(2).strip().upper()
        trip["events"].append(event)

    # Hook event
    hook_match = re.search(r"hook trailer (\d+)(?: at ([\w\s\d]+))?", command_text)
    if hook_match:
        event = {"type": "hook", "trailer": hook_match.group(1)}
        if hook_match.group(2):
            event["location"] = hook_match.group(2).strip().upper()
        trip["events"].append(event)

    # Odometer
    odo_match = re.search(r"odometer (\d+)", command_text)
    if odo_match:
        trip["odometer"] = odo_match.group(1)

    save_log(log)
    return f"✅ Trip {trip_number} updated."
