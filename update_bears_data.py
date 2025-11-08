#!/usr/bin/env python3
"""
build_reco.py

Reads the small live data files from public/ and entries/ and produces
a single recommendation file for ChatGPT:

    public/bears_reco_gw{gw}.json
    public/bears_reco_latest.json

This contains:
  - summary of Bears + Wigan squads
  - expected points for every Bears player this GW
  - baseline XI total
  - best 1FT path
  - optional best extra transfer for a -4 hit (if it actually gains points)
"""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
ENTRIES_DIR = PUBLIC_DIR / "entries"

BOOTSTRAP_PATH = PUBLIC_DIR / "bootstrap.json"
FIXTURES_PATH = PUBLIC_DIR / "fixtures.json"
META_PATH = PUBLIC_DIR / "meta.json"
FEED_PLAYERS_PATH = PUBLIC_DIR / "feed_players.csv"
RECENT_FORM_PATH = PUBLIC_DIR / "recent_form.json"

BEARS_ENTRY_PATTERN = "bears_gw{gw}.json"
WIGAN_ENTRY_PATTERN = "wigan_gw{gw}.json"

OUT_RECO_GW_PATH = PUBLIC_DIR / "bears_reco_gw{gw}.json"
OUT_RECO_LATEST_PATH = PUBLIC_DIR / "bears_reco_latest.json"


# ---------- UTILS ----------

def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ wrote {path.relative_to(BASE_DIR)}")


def _safe_float(v, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def detect_current_gw(meta: dict, bootstrap: dict) -> int:
    """
    Decide which GW we're working on, preferring meta.json.
    """
    # meta.json first if present
    for key in ("current_event", "event", "gw"):
        if key in meta and isinstance(meta[key], int):
            return meta[key]

    # fallback to bootstrap
    events = bootstrap.get("events", [])
    current = next((e for e in events if e.get("is_current")), None)
    if current:
        return current["id"]

    finished = [e for e in events if e.get("finished")]
    if finished:
        return max(finished, key=lambda e: e["id"])["id"]

    raise RuntimeError("Could not determine current gameweek")


# ---------- CORE DATA BUILDING ----------

def load_squad(entry_path: Path) -> Dict:
    """Return a simple dict with picks, captain, bank etc."""
    data = load_json(entry_path)
    picks = data.get("picks", [])
    return {
        "entry_id": data.get("entry", None),
        "bank": data.get("entry_history", {}).get("bank", 0) / 10.0,
        "team_value": data.get("entry_history", {}).get("value", 0) / 10.0,
        # this is actually "transfers used this GW", but it's fine for now
        "free_transfers": data.get("entry_history", {}).get("event_transfers", 0),
        "chip_active": data.get("active_chip", None),
        "picks": picks,
    }


def summarise_xi(squad: Dict) -> Dict:
    """Extract starting XI + bench element IDs & armband data."""
    picks = squad["picks"]
    starting = [p["element"] for p in picks if p.get("multiplier", 0) > 0]
    bench = [p["element"] for p in picks if p.get("multiplier", 0) == 0]

    captain = next((p["element"] for p in picks if p.get("is_captain")), None)
    vice = next((p["element"] for p in picks if p.get("is_vice_captain")), None)

    return {
        "bank": squad["bank"],
        "team_value": squad["team_value"],
        "free_transfers": squad["free_transfers"],
        "chip_active": squad["chip_active"],
        "captain": captain,
        "vice_captain": vice,
        "starting_xi": starting,
        "bench": bench,
    }


def build_player_lookup(feed_df: pd.DataFrame) -> Dict[int, Dict]:
    """
    Map player_id -> row dict from feed_players.csv, auto-detecting the
    correct ID column (handles 'player_id', 'id', 'element', 'code').
    """
    # auto-detect ID column
    id_col = None
    for candidate in ["player_id", "id", "element", "code"]:
        if candidate in feed_df.columns:
            id_col = candidate
            break

    if not id_col:
        raise KeyError(
            f"No valid player ID column found in feed_players.csv. "
            f"Columns = {feed_df.columns.tolist()}"
        )

    lookup: Dict[int, Dict] = {}
    for _, row in feed_df.iterrows():
        pid = int(row[id_col])
        lookup[pid] = row.to_dict()
    return lookup


def pick_best_xi(
    squad_ids: List[int],
    player_lookup: Dict[int, Dict],
    expected_points: Dict[int, float],
) -> Tuple[List[int], float]:
    """
    Greedy XI selector:
      - exactly 1 GK
      - 3–5 DEF, 2–5 MID, 1–3 FWD
      - maximise sum of expected_points
    """
    # group by position
    by_pos = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for pid in squad_ids:
        info = player_lookup.get(pid)
        if not info:
            continue
        pos = info["position"]
        by_pos.setdefault(pos, []).append(pid)

    # sort each by expected points (desc)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda pid: expected_points.get(pid, 0.0), reverse=True)

    xi: List[int] = []

    # 1 GK
    if by_pos["GK"]:
        xi.append(by_pos["GK"][0])

    # minimums: 3 DEF, 2 MID, 1 FWD
    for pos, min_needed in (("DEF", 3), ("MID", 2), ("FWD", 1)):
        xi.extend(by_pos[pos][:min_needed])

    # counts so far
    pos_counts = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}

    # remaining players pool
    remaining: List[int] = []
    for pos in ("DEF", "MID", "FWD"):
        remaining.extend(by_pos[pos][pos_counts[pos]:])

    remaining.sort(key=lambda pid: expected_points.get(pid, 0.0), reverse=True)

    # fill up to 11 respecting max limits
    max_limits = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}

    while len(xi) < 11 and remaining:
        pid = remaining.pop(0)
        pos = player_lookup[pid]["position"]
        if pos_counts[pos] < max_limits[pos]:
            xi.append(pid)
            pos_counts[pos] += 1

    total = sum(expected_points.get(pid, 0.0) for pid in xi)
    return xi, total


def compute_expected_points_for_bears(
    gw: int,
    bears_entry: Dict,
    feed_df: pd.DataFrame,
    recent_form: Dict,
) -> Tuple[Dict[int, float], Dict[int, Dict]]:
    """
    Return:
      - expected_points: player_id -> xPts for this GW
      - squad_players: player_id -> info dict (merged feed+form for Bears only)
    """
    player_lookup = build_player_lookup(feed_df)
    recent = recent_form

    squad_ids = {p["element"] for p in bears_entry["picks"]}

    expected_points: Dict[int, float] = {}
    squad_players: Dict[int, Dict] = {}

    for pid in squad_ids:
        base = player_lookup.get(pid)
        if not base:
            continue

        # recent_form keys are strings
        f = recent.get(str(pid), {}) if isinstance(recent, dict) else {}
        last5_pts = f.get("last5_points", 0)
        last5_mins = f.get("last5_minutes", 0) or 1
        last5_xgi = f.get("last5_xgi", 0.0)

        # convert to per-90 form
        form_per90 = last5_pts / (last5_mins / 90.0)

        gw_xmins = _safe_float(base.get("gw_xmins", 80.0))
        minutes_factor = gw_xmins / 90.0

        fixture_fdr = base.get("gw_fdr")
        fixture_fdr = int(fixture_fdr) if fixture_fdr is not None else 3
        # easier fixture (FDR 2) -> >1, harder (4) -> <1
        fixture_factor = 1.0 + (3 - fixture_fdr) * 0.15

        ppg = _safe_float(base.get("points_per_game", 0.0))

        xpts = (0.6 * ppg + 0.4 * form_per90 + 0.1 * last5_xgi) * minutes_factor * fixture_factor
        expected_points[pid] = xpts

        merged = dict(base)
        merged.update(
            {
                "last5_minutes": last5_mins,
                "last5_points": last5_pts,
                "last5_xgi": last5_xgi,
                "gw_xmins": gw_xmins,
                "expected_points_gw": xpts,
            }
        )
        squad_players[pid] = merged

    return expected_points, squad_players


def search_best_single_transfer(
    gw: int,
    bears_entry: Dict,
    feed_df: pd.DataFrame,
    player_lookup: Dict[int, Dict],
    expected_points: Dict[int, float],
) -> Dict:
    """
    Scan all 1-transfer moves (same position, within budget) and
    return a dict describing the best one compared to hold.
    """
    squad_ids = {p["element"] for p in bears_entry["picks"]}
    bank = bears_entry["bank"]

    # baseline
    base_xi, base_total = pick_best_xi(list(squad_ids), player_lookup, expected_points)

    best = {
        "type": "HOLD",
        "out": None,
        "in": None,
        "gain_vs_hold": 0.0,
        "new_xi": base_xi,
        "new_total": base_total,
    }

    # precompute position + price
    pos = {pid: player_lookup[pid]["position"] for pid in player_lookup}
    price = {pid: _safe_float(player_lookup[pid]["now_cost"]) for pid in player_lookup}

    # only consider realistic transfer-in pool: players with decent minutes & form
    candidates = [
        pid for pid in player_lookup
        if _safe_float(player_lookup[pid].get("gw_xmins", 0)) >= 40
    ]

    for out_pid in squad_ids:
        out_pos = pos.get(out_pid)
        if not out_pos:
            continue

        budget = bank + price.get(out_pid, 0.0)

        for in_pid in candidates:
            if in_pid in squad_ids:
                continue
            if pos.get(in_pid) != out_pos:
                continue
            if price.get(in_pid, 99) > budget:
                continue

            new_squad = list(squad_ids - {out_pid} | {in_pid})
            xi, total = pick_best_xi(new_squad, player_lookup, expected_points)
            gain = total - base_total

            if gain > best["gain_vs_hold"]:
                best = {
                    "type": "FT",
                    "out": out_pid,
                    "in": in_pid,
                    "gain_vs_hold": gain,
                    "new_xi": xi,
                    "new_total": total,
                }

    best["base_xi"] = base_xi
    best["base_total"] = base_total
    return best


def maybe_search_second_transfer_hit(
    best_ft: Dict,
    bears_entry: Dict,
    feed_df: pd.DataFrame,
    player_lookup: Dict[int, Dict],
    expected_points: Dict[int, float],
) -> Dict:
    """
    From the best 1FT result, try a second transfer for a -4 hit.
    Only keep it if (new_total - 4) improves on the best_ft.new_total.
    """
    if best_ft["type"] == "HOLD":
        base_squad_ids = {p["element"] for p in bears_entry["picks"]}
    else:
        base_squad_ids = {p["element"] for p in bears_entry["picks"]}
        base_squad_ids.remove(best_ft["out"])
        base_squad_ids.add(best_ft["in"])

    bank = bears_entry["bank"]
    # adjust bank crudely to reflect best_ft price diff
    price = {pid: _safe_float(player_lookup[pid]["now_cost"]) for pid in player_lookup}
    if best_ft["type"] == "FT":
        bank = bank + price.get(best_ft["out"], 0.0) - price.get(best_ft["in"], 0.0)

    pos = {pid: player_lookup[pid]["position"] for pid in player_lookup}
    candidates = [
        pid for pid in player_lookup
        if _safe_float(player_lookup[pid].get("gw_xmins", 0)) >= 40
    ]

    # recompute baseline from this new squad
    base_xi, base_total = pick_best_xi(list(base_squad_ids), player_lookup, expected_points)

    best_hit = {
        "type": "NONE",
        "out": None,
        "in": None,
        "gain_vs_best_ft_minus4": 0.0,
        "new_xi": best_ft["new_xi"],
        "new_total": best_ft["new_total"],
    }

    for out_pid in base_squad_ids:
        out_pos = pos.get(out_pid)
        if not out_pos:
            continue

        budget = bank + price.get(out_pid, 0.0)

        for in_pid in candidates:
            if in_pid in base_squad_ids:
                continue
            if pos.get(in_pid) != out_pos:
                continue
            if price.get(in_pid, 99) > budget:
                continue

            new_squad = list(base_squad_ids - {out_pid} | {in_pid})
            xi, total = pick_best_xi(new_squad, player_lookup, expected_points)

            # compare against best_ft.new_total with a -4 cost
            effective_gain = (total - 4.0) - best_ft["new_total"]
            if effective_gain > best_hit["gain_vs_best_ft_minus4"]:
                best_hit = {
                    "type": "HIT-4",
                    "out": out_pid,
                    "in": in_pid,
                    "gain_vs_best_ft_minus4": effective_gain,
                    "new_xi": xi,
                    "new_total": total,
                }

    # only keep if it's genuinely better
    if best_hit["type"] == "NONE" or best_hit["gain_vs_best_ft_minus4"] <= 0:
        return {
            "type": "NONE",
            "out": None,
            "in": None,
            "gain_vs_best_ft_minus4": 0.0,
            "new_xi": best_ft["new_xi"],
            "new_total": best_ft["new_total"],
        }

    return best_hit


# ---------- MAIN ----------

def main():
    print("🔄 build_reco.py starting …")

    meta = load_json(META_PATH)
    bootstrap = load_json(BOOTSTRAP_PATH)
    feed_df = pd.read_csv(FEED_PLAYERS_PATH)
    recent_form = load_json(RECENT_FORM_PATH)

    gw = detect_current_gw(meta, bootstrap)

    bears_entry_path = ENTRIES_DIR / BEARS_ENTRY_PATTERN.format(gw=gw)
    wigan_entry_path = ENTRIES_DIR / WIGAN_ENTRY_PATTERN.format(gw=gw)

    bears_entry = load_squad(bears_entry_path)
    wigan_entry = load_squad(wigan_entry_path)

    bears_summary = summarise_xi(bears_entry)
    wigan_summary = summarise_xi(wigan_entry)

    player_lookup = build_player_lookup(feed_df)

    # xPts for Bears squad
    expected_points, squad_players = compute_expected_points_for_bears(
        gw, bears_entry, feed_df, recent_form
    )

    best_ft = search_best_single_transfer(
        gw, bears_entry, feed_df, player_lookup, expected_points
    )
    best_hit = maybe_search_second_transfer_hit(
        best_ft, bears_entry, feed_df, player_lookup, expected_points
    )

    out = {
        "gw": gw,
        "generated_utc": datetime.utcnow().isoformat(),
        "bears": bears_summary,
        "wigan": wigan_summary,
        "bears_players": squad_players,          # per-player stats + xPts
        "model": {
            "baseline": {
                "xi": best_ft["base_xi"],
                "expected_points": best_ft["base_total"],
            },
            "best_ft": {
                "type": best_ft["type"],
                "out": best_ft["out"],
                "in": best_ft["in"],
                "gain_vs_hold": best_ft["gain_vs_hold"],
                "xi": best_ft["new_xi"],
                "expected_points": best_ft["new_total"],
            },
            "best_hit_minus4": {
                "type": best_hit["type"],
                "out": best_hit["out"],
                "in": best_hit["in"],
                "gain_vs_best_ft_minus4": best_hit["gain_vs_best_ft_minus4"],
                "xi": best_hit["new_xi"],
                "expected_points": best_hit["new_total"],
            },
        },
    }

    # write both GW-specific and "latest" files
    gw_path = Path(str(OUT_RECO_GW_PATH).format(gw=gw))
    save_json(gw_path, out)
    save_json(OUT_RECO_LATEST_PATH, out)

    print("🎉 build_reco.py complete")


if __name__ == "__main__":
    main()
