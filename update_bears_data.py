#!/usr/bin/env python3
"""
update_bears_data.py

Light-weight script used by the "Mirror Bears + Wigan data" workflow step.

Responsibilities:
  - Detect current gameweek from the live FPL API.
  - Download the GW picks for:
        * Birmingham Bears (your team)
        * Wigan Witches (rival)
  - Save them into public/entries as:
        public/entries/bears_gw{gw}.json
        public/entries/wigan_gw{gw}.json
  - Write a tiny meta_log.json with GW + timestamp (handy for debugging).

NO optimiser logic, NO feed_players / recent_form stuff lives here.
That’s all handled by update_model_data.py and build_reco.py.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import requests

# ------------ CONFIG ------------

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
ENTRIES_DIR = PUBLIC_DIR / "entries"

BASE_URL = "https://fantasy.premierleague.com/api"

# Your two teams
BEARS_ID = 10856343     # Birmingham Bears
WIGAN_ID = 10855167     # Wigan Witches


# ------------ HELPERS ------------

def fetch_json(url: str):
    """GET a URL and return parsed JSON (raise for HTTP errors)."""
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


def save_json(path: Path, data):
    """Write data as pretty JSON, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ wrote {path.relative_to(BASE_DIR)}")


def detect_current_gw(bootstrap: dict) -> int:
    """
    Determine the current GW from bootstrap-static:
      - prefer event with is_current = True
      - otherwise fall back to latest finished event
    """
    events = bootstrap.get("events", [])

    current = next((e for e in events if e.get("is_current")), None)
    if current:
        gw = current["id"]
        print(f"ℹ️ GW from bootstrap (is_current): {gw}")
        return gw

    finished = [e for e in events if e.get("finished")]
    if finished:
        gw = max(finished, key=lambda e: e["id"])["id"]
        print(f"ℹ️ GW from latest finished event: {gw}")
        return gw

    raise RuntimeError("Could not determine current gameweek from bootstrap-static")


# ------------ MAIN ------------

def main():
    print("🔄 update_bears_data.py starting …")

    # 1) Get bootstrap-static so we know which GW to hit
    bootstrap = fetch_json(f"{BASE_URL}/bootstrap-static/")
    gw = detect_current_gw(bootstrap)

    # 2) Fetch Bears GW picks
    bears_url = f"{BASE_URL}/entry/{BEARS_ID}/event/{gw}/picks/"
    bears_data = fetch_json(bears_url)
    bears_path = ENTRIES_DIR / f"bears_gw{gw}.json"
    save_json(bears_path, bears_data)

    # 3) Fetch Wigan GW picks
    wigan_url = f"{BASE_URL}/entry/{WIGAN_ID}/event/{gw}/picks/"
    wigan_data = fetch_json(wigan_url)
    wigan_path = ENTRIES_DIR / f"wigan_gw{gw}.json"
    save_json(wigan_path, wigan_data)

    # 4) Tiny meta log for sanity checks
    meta_log = {
        "updated_utc": datetime.utcnow().isoformat(),
        "gw": gw,
        "bears_entry_id": BEARS_ID,
        "wigan_entry_id": WIGAN_ID,
    }
    save_json(ENTRIES_DIR / "meta_log.json", meta_log)

    print("🎉 update_bears_data.py complete")


if __name__ == "__main__":
    main()
