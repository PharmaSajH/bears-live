import requests
import json
from pathlib import Path

# === TEAM IDS (these are yours) ===
BEARS_ID = 10856343        # Birmingham Bears
WIGAN_ID = 10855167        # Wigan Witches

BASE_URL = "https://fantasy.premierleague.com/api/"

public_dir = Path("public")
entries_dir = public_dir / "entries"
public_dir.mkdir(exist_ok=True)
entries_dir.mkdir(exist_ok=True)

def fetch_json(url: str):
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2))
    print(f"Saved {path}")

def main():
    # 1️⃣ Bootstrap (players, prices, teams, events)
    bootstrap = fetch_json(BASE_URL + "bootstrap-static/")
    save_json(public_dir / "bootstrap.json", bootstrap)

    # 2️⃣ Fixtures
    fixtures = fetch_json(BASE_URL + "fixtures/")
    save_json(public_dir / "fixtures.json", fixtures)

    # 3️⃣ Detect current Gameweek from bootstrap
    events = bootstrap.get("events", [])
    current = next((e for e in events if e.get("is_current")), None)
    if not current:
        print("⚠️  No current Gameweek found in bootstrap; stopping GW fetch.")
        return

    gw = current["id"]
    print(f"Detected current Gameweek: GW{gw}")

    # 4️⃣ Birmingham Bears picks
    bears_picks = fetch_json(
        BASE_URL + f"entry/{BEARS_ID}/event/{gw}/picks/"
    )
    save_json(entries_dir / f"bears_gw{gw}.json", bears_picks)

    # 5️⃣ Wigan Witches picks
    wigan_picks = fetch_json(
        BASE_URL + f"entry/{WIGAN_ID}/event/{gw}/picks/"
    )
    save_json(entries_dir / f"wigan_gw{gw}.json", wigan_picks)

if __name__ == "__main__":
    main()
