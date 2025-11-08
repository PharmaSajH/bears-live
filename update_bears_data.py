import requests, json, os
from datetime import datetime

# === CONFIG ===
BEARS_ID = 10856343      # Sajjad Hussain (Birmingham Bears)
WIGAN_ID = 10855167      # Rival (Wigan Witches)
OUTPUT_DIR = "public/entries"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Get Current Gameweek ===
meta = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
current_gw = next((e["id"] for e in meta["events"] if e["is_current"]), None)

# === Fetch Team Data ===
def fetch_team(entry_id, tag):
    r = requests.get(f"https://fantasy.premierleague.com/api/entry/{entry_id}/event/{current_gw}/picks/")
    if r.status_code == 200:
        data = r.json()
        path = f"{OUTPUT_DIR}/{tag}_gw{current_gw}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved {tag}_gw{current_gw}.json")
    else:
        print(f"⚠️ Failed to fetch data for {tag}")

fetch_team(BEARS_ID, "bears")
fetch_team(WIGAN_ID, "wigan")

# === Timestamp Log ===
log = {"updated": datetime.utcnow().isoformat()}
with open(f"{OUTPUT_DIR}/meta_log.json", "w") as f:
    json.dump(log, f, indent=2)
print("✅ Update complete.")
