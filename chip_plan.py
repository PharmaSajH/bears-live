#!/usr/bin/env python3
"""
chip_plan.py

Very lightweight, fully automated chip-planning helper.

Reads:
  - public/bootstrap.json
  - public/fixtures.json
  - public/bears_history.json
  - public/model_state.json   (optional)

Writes:
  - public/chip_plan.json

Heuristics (deliberately simple for now):
  - Detect which chips are already used.
  - Find the hardest upcoming GW (few good fixtures) → suggest Free Hit.
  - Find a "high upside" GW (many easy fixtures) → suggest Triple Captain.
  - Pick a Bench Boost GW near that.
  - Put Wildcard a few weeks before the hardest GW.
All values are suggestions only; ChatGPT will still reason on top.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict


BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"

BOOTSTRAP_PATH = PUBLIC_DIR / "bootstrap.json"
FIXTURES_PATH = PUBLIC_DIR / "fixtures.json"
BEARS_HISTORY_PATH = PUBLIC_DIR / "bears_history.json"
MODEL_STATE_PATH = PUBLIC_DIR / "model_state.json"

CHIP_PLAN_PATH = PUBLIC_DIR / "chip_plan.json"


def load_json(path: Path, default=None):
    if default is None:
        default = {}
    if not path.exists():
        print(f"⚠️ {path.relative_to(BASE_DIR)} not found, using default")
        return default
    with path.open() as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ wrote {path.relative_to(BASE_DIR)}")


def detect_current_gw_from_bootstrap(bootstrap: dict) -> int | None:
    events = bootstrap.get("events") or []
    # prefer is_current flag
    current = next((e for e in events if e.get("is_current")), None)
    if current:
        return current.get("id")
    # else use latest unfinished as "current-ish"
    unfinished = [e for e in events if not e.get("finished")]
    if unfinished:
        return min(unfinished, key=lambda e: e["
