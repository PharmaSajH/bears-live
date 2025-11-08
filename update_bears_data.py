#!/usr/bin/env python3
import json
import os
from pathlib import Path
from datetime import datetime
from time import sleep

import requests
import pandas as pd

# ------------ SAFE CONVERSION HELPERS ------------

def _safe_float(v, default=0.0):
    """Convert value to float safely (handles strings, None, bad values)."""
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v, default=0):
    """Convert value to int safely (handles strings, None, bad values)."""
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ------------ CONFIG ------------
BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
ENTRIES_DIR = PUBLIC_DIR / "entries"

BOOTSTRAP_PATH = PUBLIC_DIR / "bootstrap.json"
FIXTURES_PATH = PUBLIC_DIR / "fixtures.json"
META_PATH = PUBLIC_DIR / "meta.json"          # from mirror workflow

FEED_PLAYERS_PATH = PUBLIC_DIR / "feed_players.csv"
TEAM_STRENGTH_PATH = PUBLIC_DIR / "team_strength.json"
RECENT_FORM_PATH = PUBLIC_DIR / "recent_form.json"

BEARS_ENTRY_FILE = ENTRIES_DIR / "bears_gw11.json"
WIGAN_ENTRY_FILE = ENTRIES_DIR / "wigan_gw11.json"

BEARS_ID = 10856343
WIGAN_ID = 10855167

BASE_URL = "https://fantasy.premierleague.com/api"

# Simple rate-limit for element-summary calls
SUMMARY_SLEEP_SECONDS = 0.4   # ~2.5 req/sec, very safe


# ------------ HELPERS ------------

def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ wrote {path.relative_to(BASE_DIR)}")


def detect_current_gw(bootstrap: dict) -> int:
    """Try meta.json first, then bootstrap events, then finished events."""
    if META_PATH.exists():
        meta = load_json(META_PATH)
        for key in ("current_event", "event", "gw"):
            if key in meta and isinstance(meta[key], int):
                print(f"ℹ️ GW from meta.json: {meta[key]}")
                return meta[key]

    events = bootstrap.get("events", [])
    current = next((e for e in events if e.get("is_current")), None)
    if current:
        print(f"ℹ️ GW from bootstrap.json (is_current): {current['id']}")
        return current["id"]

    finished = [e for e in events if e.get("finished")]
    if finished:
        gw = max(finished, key=lambda e: e["id"])["id"]
        print(f"ℹ️ GW from latest finished event: {gw}")
        return gw

    raise RuntimeError("Could not determine current gameweek.")


def build_fixture_map(fixtures: list, gw: int) -> dict:
    """For a given GW, build a fixture mapping."""
    gw_fixtures = [f for f in fixtures if f.get("event") == gw]
    mapping = {}

    for f in gw_fixtures:
        home = f["team_h"]
        away = f["team_a"]
        fdr_home = f.get("team_h_difficulty")
        fdr_away = f.get("team_a_difficulty")

        mapping[home] = {"opp": away, "is_home": True, "fdr": fdr_home}
        mapping[away] = {"opp": home, "is_home": False, "fdr": fdr_away}

    return mapping


def estimate_xmins(chance_play_next):
    """Simple xMins heuristic based on chance_of_playing_* from bootstrap."""
    if chance_play_next is None:
        return 80
    try:
        c = float(chance_play_next)
    except (TypeError, ValueError):
        return 80

    if c >= 75:
        return 80
    if c >= 50:
        return 45
    if c > 0:
        return 20
    return 0


def build_feed_players(bootstrap: dict, fixtures_map: dict, gw: int) -> pd.DataFrame:
    elements = bootstrap["elements"]
    teams = bootstrap["teams"]

    df = pd.DataFrame(elements)

    # team + position
    team_lookup = {t["id"]: t["name"] for t in teams}
    df["team_name"] = df["team"].map(team_lookup)

    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    df["position"] = df["element_type"].map(pos_map)

    # GW fixture info
    df["gw_fixture"] = None
    df["gw_fdr"] = None
    df["gw_is_home"] = None

    for idx, row in df.iterrows():
        team_id = row["team"]
        info = fixtures_map.get(team_id)
        if not info:
            continue
        opp_name = team_lookup.get(info["opp"], f"Team{info['opp']}")
        suffix = "(H)" if info["is_home"] else "(A)"
        df.at[idx, "gw_fixture"] = f"{opp_name} {suffix}"
        df.at[idx, "gw_fdr"] = info["fdr"]
        df.at[idx, "gw_is_home"] = 1 if info["is_home"] else 0

    # chance_of_playing -> xMins
    if "chance_of_playing_next_round" in df.columns:
        df["chance_play_next"] = df["chance_of_playing_next_round"]
    elif "chance_of_playing_this_round" in df.columns:
        df["chance_play_next"] = df["chance_of_playing_this_round"]
    else:
        df["chance_play_next"] = None

    df["gw_xmins"] = df["chance_play_next"].apply(estimate_xmins)

    cols = {
        "id": "player_id",
        "web_name": "web_name",
        "first_name": "first_name",
        "second_name": "second_name",
        "team": "team_id",
        "team_name": "team_name",
        "position": "position",
        "now_cost": "now_cost",
        "status": "status",
        "selected_by_percent": "selected_percent",
        "form": "form",
        "points_per_game": "points_per_game",
        "total_points": "total_points",
        "minutes": "minutes_season",
        "ict_index": "ict_index",
        "influence": "influence",
        "creativity": "creativity",
        "threat": "threat",
        "chance_play_next": "chance_play_next",
        "gw_fixture": "gw_fixture",
        "gw_fdr": "gw_fdr",
        "gw_is_home": "gw_is_home",
        "gw_xmins": "gw_xmins",
    }

    out = df[list(cols.keys())].rename(columns=cols)
    out["full_name"] = out["first_name"].str.cat(out["second_name"], sep=" ")
    order = [
        "player_id", "web_name", "full_name",
        "team_id", "team_name", "position",
        "now_cost", "status", "selected_percent",
        "form", "points_per_game", "total_points", "minutes_season",
        "ict_index", "influence", "creativity", "threat",
        "chance_play_next", "gw_xmins",
        "gw_fixture", "gw_fdr", "gw_is_home",
    ]
    out = out[order]
    out["gw"] = gw
    out["generated_utc"] = datetime.utcnow().isoformat()

    return out


def build_team_strength(bootstrap: dict) -> dict:
    teams = bootstrap["teams"]
    data = []
    for t in teams:
        data.append(
            {
                "id": t["id"],
                "name": t["name"],
                "short_name": t["short_name"],
                "strength_overall_home": t.get("strength_overall_home"),
                "strength_overall_away": t.get("strength_overall_away"),
                "strength_attack_home": t.get("strength_attack_home"),
                "strength_attack_away": t.get("strength_attack_away"),
                "strength_defence_home": t.get("strength_defence_home"),
                "strength_defence_away": t.get("strength_defence_away"),
            }
        )
    return {"teams": data, "generated_utc": datetime.utcnow().isoformat()}


def fetch_json(url: str):
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


def update_entry_histories():
    """Save full entry history for Bears and Wigan."""
    bears_hist = fetch_json(f"{BASE_URL}/entry/{BEARS_ID}/history/")
    wigan_hist = fetch_json(f"{BASE_URL}/entry/{WIGAN_ID}/history/")

    save_json(PUBLIC_DIR / "bears_history.json", bears_hist)
    save_json(PUBLIC_DIR / "wigan_history.json", wigan_hist)


def get_squad_player_ids(gw: int) -> set:
    """Return all player IDs from Bears + Wigan for given GW."""
    ids = set()
    bears_path = ENTRIES_DIR / f"bears_gw{gw}.json"
    wigan_path = ENTRIES_DIR / f"wigan_gw{gw}.json"

    for path in (bears_path, wigan_path):
        if not path.exists():
            print(f"⚠️ {path.relative_to(BASE_DIR)} missing, skipping for recent_form.")
            continue
        data = load_json(path)
        for p in data.get("picks", []):
            if "element" in p:
                ids.add(p["element"])

    print(f"ℹ️ recent_form will cover {len(ids)} players (Bears + Wigan squads)")
    return ids


def build_recent_form(player_ids: set) -> dict:
    """Fetch recent 5-game stats for each player in Bears + Wigan squads."""
    recent = {}
    for i, pid in enumerate(sorted(player_ids)):
        url = f"{BASE_URL}/element-summary/{pid}/"
        data = fetch_json(url)
        history = data.get("history", [])

        # last 5 fixtures by round
        history_sorted = sorted(history, key=lambda h: h["round"])
        last5 = history_sorted[-5:]

        # use safe numeric conversion because API sometimes returns strings
        mins   = sum(_safe_int(h.get("minutes", 0)) for h in last5)
        pts    = sum(_safe_int(h.get("total_points", 0)) for h in last5)
        xgi    = sum(_safe_float(h.get("expected_goal_involvements", 0.0)) for h in last5)
        starts = sum(1 for h in last5 if h.get("started"))
        apps   = sum(1 for h in last5 if _safe_int(h.get("minutes", 0)) > 0)

        recent[pid] = {
            "last5_minutes": mins,
            "last5_points": pts,
            "last5_xgi": xgi,
            "last5_starts": starts,
            "last5_appearances": apps,
        }

        if i < len(player_ids) - 1:
            sleep(SUMMARY_SLEEP_SECONDS)

    recent["generated_utc"] = datetime.utcnow().isoformat()
    return recent


def main():
    print("🔄 update_bears_data.py starting …")

    bootstrap = load_json(BOOTSTRAP_PATH)
    fixtures_raw = load_json(FIXTURES_PATH)
    gw = detect_current_gw(bootstrap)
    fixtures_map = build_fixture_map(fixtures_raw, gw)

    # 1) feed_players.csv
    feed_df = build_feed_players(bootstrap, fixtures_map, gw)
    FEED_PLAYERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    feed_df.to_csv(FEED_PLAYERS_PATH, index=False)
    print(f"✅ wrote {FEED_PLAYERS_PATH.relative_to(BASE_DIR)}")

    # 2) team_strength.json
    team_strength = build_team_strength(bootstrap)
    save_json(TEAM_STRENGTH_PATH, team_strength)

    # 3) entry histories
    update_entry_histories()

    # 4) recent_form for Bears + Wigan squads
    player_ids = get_squad_player_ids(gw)
    if player_ids:
        recent_form = build_recent_form(player_ids)
        save_json(RECENT_FORM_PATH, recent_form)
    else:
        print("⚠️ No squad player IDs found; skipping recent_form.json")

    print("🎉 update_bears_data.py complete")


if __name__ == "__main__":
    main()
