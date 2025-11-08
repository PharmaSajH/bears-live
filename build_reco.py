#!/usr/bin/env python3
"""
build_reco.py

Generates ChatGPT recommendation JSON for Bears & Wigan.
Reads feed_players.csv and recent_form.json and outputs:

    public/bears_reco_gw{gw}.json
"""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
ENTRIES_DIR = PUBLIC_DIR / "entries"

BOOTSTRAP_PATH = PUBLIC_DIR / "bootstrap.json"
META_PATH = PUBLIC_DIR / "meta.json"
FEED_PLAYERS_PATH = PUBLIC_DIR / "feed_players.csv"
RECENT_FORM_PATH = PUBLIC_DIR / "recent_form.json"

BEARS_ENTRY_PATTERN = "bears_gw{gw}.json"
WIGAN_ENTRY_PATTERN = "wigan_gw{gw}.json"
OUT_RECO_PATH = PUBLIC_DIR / "bears_reco_gw{gw}.json"

# ---------- UTILS ----------

def load_json(path: Path):
    with path.open() as f:
        return json.load(f)

def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ wrote {path.relative_to(BASE_DIR)}")

def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default

def detect_current_gw(meta: dict, bootstrap: dict) -> int:
    for k in ("current_event", "event", "gw"):
        if isinstance(meta.get(k), int):
            return meta[k]
    events = bootstrap.get("events", [])
    current = next((e for e in events if e.get("is_current")), None)
    if current:
        return current["id"]
    finished = [e for e in events if e.get("finished")]
    if finished:
        return max(finished, key=lambda e: e["id"])["id"]
    raise RuntimeError("Cannot detect current GW")

# ---------- FIX HERE ----------
def build_player_lookup(feed_df: pd.DataFrame) -> dict:
    """
    Handles flexible CSV header naming:
      player_id / element / id  → treated equally.
    """
    cols = [c.lower().strip() for c in feed_df.columns]
    id_col = next((c for c in cols if c in ["player_id", "element", "id"]), None)
    if not id_col:
        raise KeyError("No ID column (player_id/element/id) found in feed_players.csv")

    lookup = {}
    for _, row in feed_df.iterrows():
        pid = int(row[id_col])
        lookup[pid] = row.to_dict()
    return lookup
# --------------------------------

def summarise_xi(entry_data: dict) -> dict:
    picks = entry_data.get("picks", [])
    return {
        "bank": entry_data.get("entry_history", {}).get("bank", 0) / 10.0,
        "team_value": entry_data.get("entry_history", {}).get("value", 0) / 10.0,
        "free_transfers": entry_data.get("entry_history", {}).get("event_transfers", 0),
        "chip_active": entry_data.get("active_chip"),
        "captain": next((p["element"] for p in picks if p.get("is_captain")), None),
        "vice_captain": next((p["element"] for p in picks if p.get("is_vice_captain")), None),
        "starting_xi": [p["element"] for p in picks if p.get("multiplier", 0) > 0],
        "bench": [p["element"] for p in picks if p.get("multiplier", 0) == 0],
    }

def main():
    print("🔄 build_reco.py starting …")

    meta = load_json(META_PATH)
    bootstrap = load_json(BOOTSTRAP_PATH)
    feed_df = pd.read_csv(FEED_PLAYERS_PATH)
    recent_form = load_json(RECENT_FORM_PATH)

    gw = detect_current_gw(meta, bootstrap)

    bears_entry = load_json(ENTRIES_DIR / BEARS_ENTRY_PATTERN.format(gw=gw))
    wigan_entry = load_json(ENTRIES_DIR / WIGAN_ENTRY_PATTERN.format(gw=gw))

    bears_summary = summarise_xi(bears_entry)
    wigan_summary = summarise_xi(wigan_entry)

    player_lookup = build_player_lookup(feed_df)

    out = {
        "gw": gw,
        "generated_utc": datetime.utcnow().isoformat(),
        "bears": bears_summary,
        "wigan": wigan_summary,
        "transfers": [],
        "notes": [
            "This file mirrors live FPL squads.",
            "Optimiser recommendations will populate here next."
        ]
    }

    save_json(Path(str(OUT_RECO_PATH).format(gw=gw)), out)
    print("🎉 build_reco.py complete")

if __name__ == "__main__":
    main()
