#!/usr/bin/env python3
"""
update_model_data.py

Reads core FPL data from public/ (written by update_bears_data.py)
and builds the *model-facing* helper files:

  - public/feed_players.csv          → per-player snapshot for this GW
  - public/team_strength.json        → team strength metrics
  - public/recent_form.json          → last-5-GW mins/points/xGI for Bears+Wigan squads
  - public/bears_history.json        → full entry history for Bears
  - public/wigan_history.json        → full entry history for Wigan
  - public/future_fdr.json           → future fixture difficulty summary per team
  - public/price_ownership_trend.json → time series of price & ownership for all players
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from time import sleep
from typing import Dict, List, Set

import pandas as pd
import requests


# ------------ CONFIG ------------

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

BEARS_ID = 10856343      # Birmingham Bears
WIGAN_ID = 10855167      # Wigan Witches

BASE_URL = "https://fantasy.premierleague.com/api"

SUMMARY_SLEEP_SECONDS = 0.4   # rate-limit for element-summary calls


# ------------ HELPERS ------------

def _safe_float(v, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v, default: int = 0) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


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


# ------------ GW DETECTION & FIXTURES ------------

def detect_current_gw(bootstrap: dict) -> int:
    """
    Try meta.json first (if present), otherwise bootstrap events.
    Falls back to last finished GW if nothing is marked is_current.
    """
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


def build_fixture_map(fixtures: List[dict], gw: int) -> Dict[int, dict]:
    """
    For a given GW, build:
      { team_id: {"opp": opp_id, "is_home": bool, "fdr": int} }
    """
    gw_fixtures = [f for f in fixtures if f.get("event") == gw]
    mapping: Dict[int, dict] = {}

    for f in gw_fixtures:
        home = f["team_h"]
        away = f["team_a"]
        fdr_home = f.get("team_h_difficulty")
        fdr_away = f.get("team_a_difficulty")

        mapping[home] = {"opp": away, "is_home": True,  "fdr": fdr_home}
        mapping[away] = {"opp": home, "is_home": False, "fdr": fdr_away}

    return mapping


def estimate_xmins(chance_play_next) -> float:
    """
    Simple xMins heuristic based on chance_of_playing_* from bootstrap.
    Upgradeable later.
    """
    if chance_play_next is None:
        return 80.0
    try:
        c = float(chance_play_next)
    except (TypeError, ValueError):
        return 80.0

    if c >= 75:
        return 80.0
    if c >= 50:
        return 45.0
    if c > 0:
        return 20.0
    return 0.0


# ------------ FEED PLAYERS & TEAM STRENGTH ------------

def build_feed_players(bootstrap: dict, fixtures_map: Dict[int, dict], gw: int) -> pd.DataFrame:
    """
    Build a compact per-player table (feed_players.csv) used by build_reco.py.
    Now includes injury/news fields from bootstrap.
    """
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

    # Column selection + rename to model-friendly schema
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
        # NEW: injury/news sentiment
        "news": "news",
        "news_added": "news_added",
    }

    # some older API snapshots may not have news/news_added; use intersection
    available_cols = [c for c in cols.keys() if c in df.columns]
    out = df[available_cols].rename(columns=cols)
    out["full_name"] = (
        out.get("first_name", "").astype(str).str.cat(
            out.get("second_name", "").astype(str),
            sep=" ",
        )
    )

    order = [
        "player_id", "web_name", "full_name",
        "team_id", "team_name", "position",
        "now_cost", "status", "selected_percent",
        "form", "points_per_game", "total_points", "minutes_season",
        "ict_index", "influence", "creativity", "threat",
        "chance_play_next", "gw_xmins",
        "gw_fixture", "gw_fdr", "gw_is_home",
    ]
    # keep injury/news at the end if present
    if "news" in out.columns:
        order.append("news")
    if "news_added" in out.columns:
        order.append("news_added")

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


# ------------ FUTURE FIXTURE DIFFICULTY ------------

def build_future_fdr(bootstrap: dict, fixtures: List[dict], current_gw: int, horizon: int = 8) -> dict:
    """
    For each team, summarise the next `horizon` GWs of fixture difficulty.
    """
    teams = {t["id"]: t for t in bootstrap.get("teams", [])}
    upper_gw = current_gw + horizon

    per_team = {}

    for t_id, t in teams.items():
        per_team[t_id] = {
            "team_id": t_id,
            "team_name": t["name"],
            "short_name": t["short_name"],
            "fixtures": [],
        }

    for f in fixtures:
        gw = f.get("event")
        if gw is None or gw < current_gw or gw > upper_gw:
            continue

        home = f["team_h"]
        away = f["team_a"]
        fdr_home = f.get("team_h_difficulty")
        fdr_away = f.get("team_a_difficulty")

        # home side
        per_team[home]["fixtures"].append(
            {
                "gw": gw,
                "is_home": True,
                "opp_team_id": away,
                "opp_team_name": teams.get(away, {}).get("name"),
                "fdr": fdr_home,
            }
        )
        # away side
        per_team[away]["fixtures"].append(
            {
                "gw": gw,
                "is_home": False,
                "opp_team_id": home,
                "opp_team_name": teams.get(home, {}).get("name"),
                "fdr": fdr_away,
            }
        )

    # summary stats
    for t_id, rec in per_team.items():
        fs = rec["fixtures"]
        if not fs:
            rec["avg_fdr"] = None
            rec["easy_fixtures"] = 0
            rec["hard_fixtures"] = 0
            continue
        fdr_vals = [f["fdr"] for f in fs if f["fdr"] is not None]
        rec["avg_fdr"] = sum(fdr_vals) / len(fdr_vals) if fdr_vals else None
        rec["easy_fixtures"] = sum(1 for f in fs if f["fdr"] is not None and f["fdr"] <= 2)
        rec["hard_fixtures"] = sum(1 for f in fs if f["fdr"] is not None and f["fdr"] >= 4)

    return {
        "gw_start": current_gw,
        "gw_end": upper_gw,
        "teams": list(per_team.values()),
        "generated_utc": datetime.utcnow().isoformat(),
    }


# ------------ ENTRY HISTORIES + RECENT FORM ------------

def fetch_json(url: str):
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


def update_entry_histories():
    """Save full entry history for Bears and Wigan."""
    bears_hist = fetch_json(f"{BASE_URL}/entry/{BEARS_ID}/history/")
    wigan_hist = fetch_json(f"{BASE_URL}/entry/{WIGAN_ID}/history/")

    save_json(BEARS_HISTORY_PATH, bears_hist)
    save_json(WIGAN_HISTORY_PATH, wigan_hist)


def get_squad_player_ids(gw: int) -> Set[int]:
    """
    Read bears_gw{gw}.json and wigan_gw{gw}.json from public/entries
    and return the union of all picked element IDs.
    """
    ids: Set[int] = set()
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


def build_recent_form(player_ids: Set[int]) -> dict:
    """
    For each player_id in player_ids, call /element-summary/{id}/
    and summarise last 5 GWs (mins, points, xGI).
    """
    recent: dict = {}
    for i, pid in enumerate(sorted(player_ids)):
        url = f"{BASE_URL}/element-summary/{pid}/"
        data = fetch_json(url)
        history = data.get("history", [])

        history_sorted = sorted(history, key=lambda h: h["round"])
        last5 = history_sorted[-5:]

        mins = sum(_safe_int(h.get("minutes", 0)) for h in last5)
        pts = sum(_safe_int(h.get("total_points", 0)) for h in last5)
        xgi = sum(_safe_float(h.get("expected_goal_involvements", 0.0)) for h in last5)
        starts = sum(1 for h in last5 if h.get("started"))
        apps = sum(1 for h in last5 if _safe_int(h.get("minutes", 0)) > 0)

        recent[str(pid)] = {
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


# ------------ PRICE & OWNERSHIP TREND ------------

def update_price_ownership_trend(feed_df: pd.DataFrame, gw: int):
    """
    Append a snapshot of price & ownership for ALL players into
    price_ownership_trend.json.
    """
    now = datetime.utcnow().isoformat()

    # Detect player_id column from feed_df
    pid_col_candidates = [c for c in feed_df.columns
                          if str(c).lower() in ("player_id", "id", "element", "code")]
    if not pid_col_candidates:
        print("⚠️ Could not find player_id column for trend; skipping.")
        return

    pid_col = pid_col_candidates[0]

    selected_col = "selected_percent" if "selected_percent" in feed_df.columns else None
    if selected_col is None:
        print("⚠️ No selected_percent column found; skipping ownership trend.")
        return

    if "now_cost" not in feed_df.columns:
        print("⚠️ No now_cost column found; skipping price trend.")
        return

    trend = load_json(PRICE_OWNERSHIP_TREND_PATH, default={"records": []})
    records = trend.get("records", [])

    for _, row in feed_df.iterrows():
        pid = _safe_int(row[pid_col])
        price = _safe_float(row["now_cost"]) / 10.0  # convert to millions
        sel = _safe_float(row[selected_col])

        records.append(
            {
                "timestamp_utc": now,
                "gw": gw,
                "player_id": pid,
                "now_cost": price,
                "selected_percent": sel,
            }
        )

    # optional: cap records length to avoid unbounded growth
    if len(records) > 200000:
        records = records[-200000:]

    trend["records"] = records
    trend["generated_utc"] = now
    save_json(PRICE_OWNERSHIP_TREND_PATH, trend)


# ------------ MAIN ------------

def main():
    print("🔄 update_model_data.py starting …")

    bootstrap = load_json(BOOTSTRAP_PATH)
    fixtures_raw = load_json(FIXTURES_PATH, default=[])
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

    # 3) future_fdr.json (next 8 GWs by default)
    future_fdr = build_future_fdr(bootstrap, fixtures_raw, gw, horizon=8)
    save_json(FUTURE_FDR_PATH, future_fdr)

    # 4) entry histories
    update_entry_histories()

    # 5) recent_form for Bears + Wigan squads
    player_ids = get_squad_player_ids(gw)
    if player_ids:
        recent_form = build_recent_form(player_ids)
        save_json(RECENT_FORM_PATH, recent_form)
    else:
        print("⚠️ No squad player IDs found; skipping recent_form.json")

    # 6) price & ownership trend snapshot
    update_price_ownership_trend(feed_df, gw)

    print("🎉 update_model_data.py complete")


if __name__ == "__main__":
    main()
