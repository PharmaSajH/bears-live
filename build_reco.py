#!/usr/bin/env python3
"""
build_reco.py

Simple recommendation scaffold for Birmingham Bears vs Wigan Witches.

This script:
  - Detects current GW from meta.json + bootstrap.json
  - Reads Bears and Wigan GW picks (bears_gw{gw}.json / wigan_gw{gw}.json)
  - Summarises bank, team value, captain/vice, XI & bench
  - Writes:
        public/bears_reco_gw{gw}.json   (per-GW snapshot)
        public/bears_reco_latest.json   (always latest snapshot)

At this stage, it does NOT run a full optimiser; it just mirrors
live squads in a stable, machine-readable format for ChatGPT.
"""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# ---------- CONFIG ----------

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
ENTRIES_DIR = PUBLIC_DIR / "entries"

BOOTSTRAP_PATH = PUBLIC_DIR / "bootstrap.json"
META_PATH = PUBLIC_DIR / "meta.json"

BEARS_ENTRY_PATTERN = "bears_gw{gw}.json"
WIGAN_ENTRY_PATTERN = "wigan_gw{gw}.json"

OUT_RECO_GW_PATTERN = "bears_reco_gw{gw}.json"
OUT_RECO_LATEST = "bears_reco_latest.json"


# ---------- UTILS ----------

def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ wrote {path.relative_to(BASE_DIR)}")


def detect_current_gw(meta: dict, bootstrap: dict) -> int:
    """
    Decide which GW we're working on, preferring meta.json if present.
    """
    # meta.json first if present
    for key in ("current_event", "event", "gw"):
        if key in meta and isinstance(meta[key], int):
            gw = meta[key]
            print(f"ℹ️ GW from meta.json: {gw}")
            return gw

    # fallback to bootstrap
    events = bootstrap.get("events", [])
    current = next((e for e in events if e.get("is_current")), None)
    if current:
        gw = current["id"]
        print(f"ℹ️ GW from bootstrap.json (is_current): {gw}")
        return gw

    finished = [e for e in events if e.get("finished")]
    if finished:
        gw = max(finished, key=lambda e: e["id"])["id"]
        print(f"ℹ️ GW from latest finished event: {gw}")
        return gw

    raise RuntimeError("Could not determine current gameweek.")


def summarise_xi(entry_data: dict) -> Dict[str, Any]:
    """
    Extract bank, value, FT, chip, captain/vice, starting XI & bench
    from a raw /entry/{id}/event/{gw}/picks/ payload.
    """
    picks = entry_data.get("picks", [])
    entry_hist = entry_data.get("entry_history", {}) or {}

    starting = [p["element"] for p in picks if p.get("multiplier", 0) > 0]
    bench = [p["element"] for p in picks if p.get("multiplier", 0) == 0]

    captain = next((p["element"] for p in picks if p.get("is_captain")), None)
    vice = next((p["element"] for p in picks if p.get("is_vice_captain")), None)

    return {
        "bank": entry_hist.get("bank", 0) / 10.0,
        "team_value": entry_hist.get("value", 0) / 10.0,
        # NOTE: FPL API's event_transfers is "transfers used" not "FTs",
        # but we're just logging it; real FT logic lives inside FPL itself.
        "free_transfers": entry_hist.get("event_transfers", 0),
        "chip_active": entry_data.get("active_chip"),
        "captain": captain,
        "vice_captain": vice,
        "starting_xi": starting,
        "bench": bench,
    }


# ---------- MAIN ----------

def main() -> None:
    print("🔄 build_reco.py starting …")

    # Load core metadata
    if META_PATH.exists():
        meta = load_json(META_PATH)
    else:
        meta = {}

    bootstrap = load_json(BOOTSTRAP_PATH)

    gw = detect_current_gw(meta, bootstrap)
    print(f"➡️  Using GW{gw} for reco")

    # Load raw entry data for Bears & Wigan
    bears_entry_path = ENTRIES_DIR / BEARS_ENTRY_PATTERN.format(gw=gw)
    wigan_entry_path = ENTRIES_DIR / WIGAN_ENTRY_PATTERN.format(gw=gw)

    if not bears_entry_path.exists():
        raise FileNotFoundError(f"Bears entry file missing: {bears_entry_path}")
    if not wigan_entry_path.exists():
        raise FileNotFoundError(f"Wigan entry file missing: {wigan_entry_path}")

    bears_entry = load_json(bears_entry_path)
    wigan_entry = load_json(wigan_entry_path)

    # Summarise squads into a compact structure
    bears_summary = summarise_xi(bears_entry)
    wigan_summary = summarise_xi(wigan_entry)

    out = {
        "gw": gw,
        "generated_utc": datetime.utcnow().isoformat(),
        "bears": bears_summary,
        "wigan": wigan_summary,
        "transfers": [],   # placeholder for future optimiser output
        "notes": [
            "This file mirrors live FPL squads for Birmingham Bears and Wigan Witches.",
            "Later versions can add optimiser recommendations into the 'transfers' field."
        ],
    }

    # Write per-GW file
    gw_filename = OUT_RECO_GW_PATTERN.format(gw=gw)
    out_path_gw = PUBLIC_DIR / gw_filename
    save_json(out_path_gw, out)

    # Write stable 'latest' alias
    out_path_latest = PUBLIC_DIR / OUT_RECO_LATEST
    save_json(out_path_latest, out)

    print(f"🎉 build_reco.py complete for GW{gw}")


if __name__ == "__main__":
    main()
