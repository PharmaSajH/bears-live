#!/usr/bin/env python3
"""
build_reco.py – Full Birmingham Bears optimiser (v2: fixes team name bug)
"""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import pandas as pd

# ---------- CONFIG ----------

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
ENTRIES_DIR = PUBLIC_DIR / "entries"

BOOTSTRAP_PATH = PUBLIC_DIR / "bootstrap.json"
META_PATH = PUBLIC_DIR / "meta.json"
FIXTURES_PATH = PUBLIC_DIR / "fixtures.json"
FEED_PLAYERS_PATH = PUBLIC_DIR / "feed_players.csv"
RECENT_FORM_PATH = PUBLIC_DIR / "recent_form.json"

BEARS_ENTRY_PATTERN = "bears_gw{gw}.json"
WIGAN_ENTRY_PATTERN = "wigan_gw{gw}.json"

OUT_RECO_GW_PATTERN = "bears_reco_gw{gw}.json"
OUT_RECO_LATEST = "bears_reco_latest.json"


# ---------- UTILITIES ----------

def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)

def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ wrote {path.relative_to(BASE_DIR)}")

def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default

def _safe_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default

def detect_current_gw(meta: dict, bootstrap: dict) -> int:
    for key in ("current_event", "event", "gw"):
        if key in meta and isinstance(meta[key], int):
            return meta[key]
    for e in bootstrap.get("events", []):
        if e.get("is_current"):
            return e["id"]
    raise RuntimeError("Could not determine current gameweek.")


# ---------- PLAYER LOOKUP ----------

def build_player_lookup(feed_df: pd.DataFrame) -> Dict[int, Dict]:
    id_col = next((c for c in ["player_id", "id", "element", "code"] if c in feed_df.columns), None)
    if not id_col:
        raise KeyError("No ID column found in feed_players.csv")
    return {int(r[id_col]): r.to_dict() for _, r in feed_df.iterrows()}

def detect_team_key(sample_row: dict) -> str:
    for c in ["team_id", "team", "teamid"]:
        if c in sample_row:
            return c
    raise KeyError("No team field found")


# ---------- FIXTURE SCORING ----------

def build_multi_gw_fixture_scores(fixtures: list, gw: int, horizon: int = 3) -> Dict[str, float]:
    """Handle both numeric IDs and team names gracefully."""
    window = set(range(gw, gw + horizon))
    total: Dict[str, float] = {}
    count: Dict[str, int] = {}

    for f in fixtures:
        if f.get("event") not in window:
            continue

        for side in [("team_h", "team_h_difficulty"), ("team_a", "team_a_difficulty")]:
            team = str(f.get(side[0]))
            fdr = _safe_int(f.get(side[1]), 3)
            score = 6 - fdr  # invert FDR: lower = better
            total[team] = total.get(team, 0) + score
            count[team] = count.get(team, 0) + 1

    return {k: total[k] / max(count[k], 1) for k in total}


# ---------- EXPECTED POINTS ----------

def compute_expected_points_all(feed_df, recent_form, multi_fix_scores, team_key):
    lookup = build_player_lookup(feed_df)
    out = {}
    for pid, base in lookup.items():
        rf = recent_form.get(str(pid), {}) if isinstance(recent_form, dict) else {}
        last5_pts = rf.get("last5_points", 0)
        last5_mins = rf.get("last5_minutes", 0)
        last5_xgi = rf.get("last5_xgi", 0.0)

        if last5_mins > 0:
            form_per90 = last5_pts / (last5_mins / 90.0)
        else:
            form_per90 = _safe_float(base.get("points_per_game", 0))

        gw_xmins = _safe_float(base.get("gw_xmins", 80))
        fixture_fdr = _safe_int(base.get("gw_fdr"), 3)
        fixture_factor = 1.0 + (3 - fixture_fdr) * 0.15

        ppg = _safe_float(base.get("points_per_game", 0))
        base_xpts = (0.6 * ppg + 0.4 * form_per90 + 0.1 * last5_xgi) * (gw_xmins / 90.0) * fixture_factor

        team_id = str(base.get(team_key))
        multi_factor = 1.0
        if multi_fix_scores:
            multi_factor = 1.0 + 0.05 * (multi_fix_scores.get(team_id, 3.0) - 3.0)

        out[pid] = base_xpts * multi_factor
    return out


# ---------- MAIN ----------

def main():
    print("🔄 build_reco.py starting …")

    meta = load_json(META_PATH) if META_PATH.exists() else {}
    bootstrap = load_json(BOOTSTRAP_PATH)
    fixtures = load_json(FIXTURES_PATH) if FIXTURES_PATH.exists() else []
    feed_df = pd.read_csv(FEED_PLAYERS_PATH)
    recent_form = load_json(RECENT_FORM_PATH) if RECENT_FORM_PATH.exists() else {}

    gw = detect_current_gw(meta, bootstrap)
    print(f"Using GW{gw} for reco")

    player_lookup = build_player_lookup(feed_df)
    sample_row = next(iter(player_lookup.values()))
    team_key = detect_team_key(sample_row)

    multi_fix_scores = build_multi_gw_fixture_scores(fixtures, gw) if fixtures else None
    expected_points = compute_expected_points_all(feed_df, recent_form, multi_fix_scores, team_key)

    # For now we just write basic sanity check
    out_path = PUBLIC_DIR / f"bears_reco_gw{gw}.json"
    save_json(out_path, {"gw": gw, "players_count": len(expected_points)})
    save_json(PUBLIC_DIR / "bears_reco_latest.json", {"gw": gw, "players_count": len(expected_points)})
    print("🎉 build_reco.py complete")

if __name__ == "__main__":
    main()
