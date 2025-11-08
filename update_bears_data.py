#!/usr/bin/env python3
"""
update_bears_data.py

Mirror live FPL data into the repo so other scripts can build the model.

Writes into public/:
  - bootstrap.json        (full FPL bootstrap)
  - fixtures.json         (all fixtures with FDR)
  - meta.json             (current GW + timestamp)
  - entries/bears_gw{gw}.json
  - entries/wigan_gw{gw}.json
  - live_gw{gw}.json      (BEST-EFFORT: live event data for the current GW)
"""

import json
import os
from datetime import datetime
from pathlib import Path

import requests

# --- CONFIG ---
BEARS_ID = 10856343      # Birmingham Bears
WIGAN_ID = 10855167      # Wigan Witches

BASE_URL = "https://fantasy.premierleague.com/api"

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
ENTRIES_DIR = PUBLIC_DIR / "entries"

os.makedirs(ENTRIES_DIR, exist_ok=True)


def fetch_json(url: str):
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


def save(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {path.relative_to(BASE_DIR)}")


def detect_current_gw(bootstrap: dict) -> int:
    events = bootstrap.get("events", [])
    current = next((e for e in events if e.get("is_current")), None)
    if current:
        return current["id"]

    # fallback: earliest unfinished
    unfinished = [e for e in events if not e.get("finished")]
    if unfinished:
        return min(unfinished, key=lambda e: e["id"])["id"]

    # last resort: max id
    if events:
        return max(events, key=lambda e: e["id"])["id"]

    raise RuntimeError("Could not determine current GW from bootstrap")


def main():
    # 1) bootstrap & fixtures
    bootstrap = fetch_json(f"{BASE_URL}/bootstrap-static/")
    fixtures = fetch_json(f"{BASE_URL}/fixtures/")

    gw = detect_current_gw(bootstrap)
    print(f"ℹ️ Current GW (for mirroring): {gw}")

    save(PUBLIC_DIR / "bootstrap.json", bootstrap)
    save(PUBLIC_DIR / "fixtures.json", fixtures)

    # 2) entries picks for Bears & Wigan for this GW
    bears = fetch_json(f"{BASE_URL}/entry/{BEARS_ID}/event/{gw}/picks/")
    wigan = fetch_json(f"{BASE_URL}/entry/{WIGAN_ID}/event/{gw}/picks/")

    save(ENTRIES_DIR / f"bears_gw{gw}.json", bears)
    save(ENTRIES_DIR / f"wigan_gw{gw}.json", wigan)

    # 3) BEST-EFFORT: live GW event data for calibration
    #    This will only exist once the GW has at least some matches played.
    try:
        live = fetch_json(f"{BASE_URL}/event/{gw}/live/")
        save(PUBLIC_DIR / f"live_gw{gw}.json", live)
        live_flag = True
    except Exception as e:
        print(f"⚠️ Could not fetch live data for GW{gw}: {e}")
        live_flag = False

    # 4) simple meta record for other scripts
    meta = {
        "updated_utc": datetime.utcnow().isoformat(),
        "gw": gw,
        "current_event": gw,
        "live_available": live_flag,
    }
    save(PUBLIC_DIR / "meta.json", meta)

    print("🎉 update_bears_data.py complete")


if __name__ == "__main__":
    main()
