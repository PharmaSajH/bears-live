import requests
import json
import os
from datetime import datetime

# --- CONFIG ---
BEARS_ID = 10856343      # Birmingham Bears
WIGAN_ID = 10855167      # Wigan Witches

BASE_URL = "https://fantasy.premierleague.com/api"
OUTPUT_DIR = "public/entries"

# make sure the folder exists on every run
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_json(url: str):
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def save(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {path}")

def main():
    # 1) get bootstrap so we know the current GW
    bootstrap = fetch_json(f"{BASE_URL}/bootstrap-static/")
    events = bootstrap.get("events", [])
    current = next((e for e in events if e.get("is_current")), None)
    if not current:
        print("⚠️ No current gameweek found")
        return
    gw = current["id"]
    print(f"Current GW: {gw}")

    # 2) save your team
    bears = fetch_json(f"{BASE_URL}/entry/{BEARS_ID}/event/{gw}/picks/")
    save(os.path.join(OUTPUT_DIR, f"bears_gw{gw}.json"), bears)

    # 3) save Wigan Witches
    wigan = fetch_json(f"{BASE_URL}/entry/{WIGAN_ID}/event/{gw}/picks/")
    save(os.path.join(OUTPUT_DIR, f"wigan_gw{gw}.json"), wigan)

    # 4) simple log
    meta = {"updated_utc": datetime.utcnow().isoformat(), "gw": gw}
    save(os.path.join(OUTPUT_DIR, "meta_log.json"), meta)

if __name__ == "__main__":
    main()
