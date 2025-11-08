#!/usr/bin/env python3
"""
build_full_snapshot.py

Combine all the small FPL data files in public/ into a single
snapshot JSON for ChatGPT:

    public/fpl_full_snapshot.json

Snapshot includes:
  - meta              → GW, timestamps, live_available
  - bootstrap         → full /bootstrap-static/
  - fixtures          → full /fixtures/ list
  - bears_entry       → /entry/{id}/event/{gw}/picks/ for Birmingham Bears
  - wigan_entry       → /entry/{id}/event/{gw}/picks/ for Wigan Witches
  - feed_players      → row-per-player table from feed_players.csv
  - team_strength     → aggregated team strength metrics
  - recent_form       → last-5-GW mins/points/xGI for Bears+Wigan squads
  - future_fdr        → future fixture difficulty (next ~8 GWs)
  - price_ownership_trend → history of price/ownership snapshots
  - bears_history     → full entry history for Bears
  - wigan_history     → full entry history for Wigan
  - live_event        → raw /event/{gw}/live/ (if available)
  - live_score_summary → simple list of current GW fixtures & scores

You can upload *only this one file* to ChatGPT and it will have
all the data it needs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


# ------------ PATHS / CONFIG ------------

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
ENTRIES_DIR = PUBLIC_DIR / "entries"

BOOTSTRAP_PATH = PUBLIC_DIR / "bootstrap.json"
FIXTURES_PATH = PUBLIC_DIR / "fixtures.json"
META_PATH = PUBLIC_DIR / "meta.json"

FEED_PLAYERS_PATH = PUBLIC_DIR / "feed_players.csv"
TEAM_STRENGTH_PATH = PUBLIC_DIR / "team_strength.json"
RECENT_FORM_PATH = PUBLIC_DIR / "recent_form.json"
FUTURE_FDR_PATH = PUBLIC_DIR / "future_fdr.json"
PRICE_OWNERSHIP_TREND_PATH = PUBLIC_DIR / "price_ownership_trend.json"

BEARS_HISTORY_PATH = PUBLIC_DIR / "bears_history.json"
WIGAN_HISTORY_PATH = PUBLIC_DIR / "wigan_history.json"

SNAPSHOT_PATH = PUBLIC_DIR / "fpl_full_snapshot.json"


# ------------ HELPERS ------------

def load_json(path: Path, default: Any = None) -> Any:
    """Safe JSON loader with default + logging."""
    if default is None:
        default = {}
    if not path.exists():
        print(f"⚠️ {path.relative_to(BASE_DIR)} not found, using default")
        return default
    with path.open() as f:
        return json.load(f)


def save_json(path: Path, data: Any):
    """Write JSON with pretty indent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ wrote {path.relative_to(BASE_DIR)}")


def detect_current_gw(meta: Dict, bootstrap: Dict) -> int:
    """
    Determine current GW, preferring meta.json but falling back to bootstrap events.
    """
    # 1) meta.json
    for key in ("current_event", "gw", "event"):
        if key in meta and isinstance(meta[key], int):
            print(f"ℹ️ GW from meta.json: {meta[key]}")
            return meta[key]

    # 2) bootstrap events
    events = bootstrap.get("events", [])
    current = next((e for e in events if e.get("is_current")), None)
    if current:
        print(f"ℹ️ GW from bootstrap (is_current): {current['id']}")
        return current["id"]

    finished = [e for e in events if e.get("finished")]
    if finished:
        gw = max(finished, key=lambda e: e["id"])["id"]
        print(f"ℹ️ GW from latest finished event: {gw}")
        return gw

    raise RuntimeError("Could not determine current gameweek")


def load_feed_players(path: Path) -> List[Dict]:
    """Load feed_players.csv as a list of dict rows."""
    if not path.exists():
        print(f"⚠️ {path.relative_to(BASE_DIR)} missing, feed_players will be empty")
        return []
    df = pd.read_csv(path)
    records = df.to_dict(orient="records")
    print(f"ℹ️ feed_players: {len(records)} records")
    return records


def load_entry_for_gw(entry_prefix: str, gw: int) -> Dict:
    """
    Load an entry picks file, e.g. entries/bears_gw{gw}.json.
    entry_prefix: 'bears' or 'wigan'.
    """
    path = ENTRIES_DIR / f"{entry_prefix}_gw{gw}.json"
    data = load_json(path, default={})
    if data:
        print(f"ℹ️ Loaded {entry_prefix}_gw{gw}.json")
    else:
        print(f"⚠️ {entry_prefix}_gw{gw}.json missing or empty")
    return data


def build_live_score_summary(
    bootstrap: Dict,
    fixtures: List[Dict],
    gw: int,
) -> List[Dict]:
    """
    Build a simple list of current-GW fixtures + scores.

    Uses fixtures.json, which FPL updates with team_h_score/team_a_score and finished flags.
    """
    teams = {t["id"]: t for t in bootstrap.get("teams", [])}
    gw_fixtures = [f for f in fixtures if f.get("event") == gw]

    summary: List[Dict] = []

    for f in gw_fixtures:
        home_id = f.get("team_h")
        away_id = f.get("team_a")
        home = teams.get(home_id, {})
        away = teams.get(away_id, {})

        # status derivation
        finished = f.get("finished")
        started = f.get("started")
        if finished:
            status = "FINISHED"
        elif started:
            status = "IN_PLAY"
        else:
            status = "SCHEDULED"

        summary.append(
            {
                "fixture_id": f.get("id"),
                "gw": gw,
                "kickoff_time": f.get("kickoff_time"),
                "status": status,
                "finished": finished,
                "started": started,
                "home_team_id": home_id,
                "home_team": home.get("name"),
                "home_short_name": home.get("short_name"),
                "home_goals": f.get("team_h_score"),
                "away_team_id": away_id,
                "away_team": away.get("name"),
                "away_short_name": away.get("short_name"),
                "away_goals": f.get("team_a_score"),
                "fdr_home": f.get("team_h_difficulty"),
                "fdr_away": f.get("team_a_difficulty"),
            }
        )

    print(f"ℹ️ live_score_summary: {len(summary)} fixtures for GW{gw}")
    return summary


# ------------ MAIN ------------

def main():
    print("🔄 build_full_snapshot.py starting …")

    # Core inputs
    bootstrap = load_json(BOOTSTRAP_PATH, default={})
    fixtures = load_json(FIXTURES_PATH, default=[])
    meta = load_json(META_PATH, default={})

    gw = detect_current_gw(meta, bootstrap)

    # Bears / Wigan entries for this GW
    bears_entry = load_entry_for_gw("bears", gw)
    wigan_entry = load_entry_for_gw("wigan", gw)

    # Tabular / helper data
    feed_players = load_feed_players(FEED_PLAYERS_PATH)
    team_strength = load_json(TEAM_STRENGTH_PATH, default={})
    recent_form = load_json(RECENT_FORM_PATH, default={})
    future_fdr = load_json(FUTURE_FDR_PATH, default={})
    price_ownership_trend = load_json(PRICE_OWNERSHIP_TREND_PATH, default={})
    bears_history = load_json(BEARS_HISTORY_PATH, default={})
    wigan_history = load_json(WIGAN_HISTORY_PATH, default={})

    # Live event (best-effort) for this GW
    live_path = PUBLIC_DIR / f"live_gw{gw}.json"
    live_event = load_json(live_path, default=None) if live_path.exists() else None
    live_available = bool(live_event)

    # Current GW scores summary from fixtures.json
    live_score_summary = build_live_score_summary(bootstrap, fixtures, gw)

    # Build snapshot
    snapshot = {
        "meta": {
            "gw": gw,
            "generated_utc": datetime.utcnow().isoformat(),
            "live_available": live_available,
        },
        "bootstrap": bootstrap,
        "fixtures": fixtures,
        "bears_entry": bears_entry,
        "wigan_entry": wigan_entry,
        "feed_players": feed_players,
        "team_strength": team_strength,
        "recent_form": recent_form,
        "future_fdr": future_fdr,
        "price_ownership_trend": price_ownership_trend,
        "bears_history": bears_history,
        "wigan_history": wigan_history,
        "live_event": live_event,
        "live_score_summary": live_score_summary,
    }

    save_json(SNAPSHOT_PATH, snapshot)
    print("🎉 build_full_snapshot.py complete")


if __name__ == "__main__":
    main()
