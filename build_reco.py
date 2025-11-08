#!/usr/bin/env python3
"""
build_reco.py

Reads the small live data files from public/ and entries/ and produces
a single recommendation file for ChatGPT:

    public/bears_reco_gw{gw}.json
    public/bears_reco_latest.json

Contents:
  - summary of Bears + Wigan squads
  - expected points for every Bears player this GW
  - baseline XI total
  - best 1FT path
  - best extra transfer for a -4 hit (only if it genuinely improves)
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set

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

OUT_RECO_TEMPLATE = "public/bears_reco_gw{gw}.json"
OUT_RECO_LATEST = PUBLIC_DIR / "bears_reco_latest.json"


# ---------- BASIC UTILS ----------


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


def _safe_int(v, default: int = 0) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ---------- GW DETECTION ----------


def detect_current_gw(meta: dict, bootstrap: dict) -> int:
    """
    Work out which GW to use.

    Priority:
      1) meta.json: "current_event" / "event" / "gw"
      2) bootstrap.events where is_current = True
      3) latest finished event
    """
    for key in ("current_event", "event", "gw"):
        if key in meta and isinstance(meta[key], int):
            print(f"ℹ️ GW from meta.json: {meta[key]}")
            return meta[key]

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


# ---------- SQUAD LOADING / SUMMARY ----------


def load_squad(entry_path: Path) -> Dict:
    """Return a simple dict with picks, captain, bank etc."""
    data = load_json(entry_path)
    picks = data.get("picks", [])
    entry_hist = data.get("entry_history", {}) or {}

    return {
        "entry_id": data.get("entry", None),
        "bank": entry_hist.get("bank", 0) / 10.0,   # millions
        "team_value": entry_hist.get("value", 0) / 10.0,
        "free_transfers": entry_hist.get("event_transfers", 0),
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


# ---------- PLAYER LOOKUPS ----------


def build_player_lookup(feed_df: pd.DataFrame) -> Dict[int, Dict]:
    """Map player_id -> row dict from feed_players.csv."""
    lookup: Dict[int, Dict] = {}
    if "player_id" not in feed_df.columns:
        raise KeyError("feed_players.csv is missing 'player_id' column")

    for _, row in feed_df.iterrows():
        pid = int(row["player_id"])
        lookup[pid] = row.to_dict()
    return lookup


def _price_millions(raw_now_cost) -> float:
    """
    FPL now_cost is usually an int (e.g. 74 for 7.4m).
    If it's already a small float (<= 20) we'll assume it's in millions.
    """
    v = _safe_float(raw_now_cost, 0.0)
    if v == 0:
        return 0.0
    if v <= 20:
        return v
    return v / 10.0


# ---------- EXPECTED POINTS MODEL ----------


def compute_expected_points_all(
    feed_df: pd.DataFrame,
    recent_form: Dict,
) -> Tuple[Dict[int, float], Dict[int, Dict]]:
    """
    Compute expected points for *all* players in feed_players.csv.

    Returns:
      expected_points: player_id -> xPts for this GW
      players_info:    player_id -> merged feed+form dict
    """
    expected_points: Dict[int, float] = {}
    players_info: Dict[int, Dict] = {}

    for _, row in feed_df.iterrows():
        pid = int(row["player_id"])

        # recent form block (last 5 GWs)
        r = recent_form.get(str(pid), {}) if isinstance(recent_form, dict) else {}
        last5_pts = _safe_float(r.get("last5_points", 0.0))
        last5_mins = _safe_float(r.get("last5_minutes", 0.0))
        last5_xgi = _safe_float(r.get("last5_xgi", 0.0))

        if last5_mins <= 0:
            form_per90 = 0.0
        else:
            form_per90 = last5_pts * 90.0 / last5_mins

        # season base
        ppg = _safe_float(row.get("points_per_game", 0.0))
        gw_xmins = _safe_float(row.get("gw_xmins", 80.0))
        minutes_factor = gw_xmins / 90.0

        # fixture difficulty (3 is neutral)
        gw_fdr = row.get("gw_fdr")
        if gw_fdr is None:
            fixture_factor = 1.0
        else:
            fdr = _safe_float(gw_fdr, 3.0)
            # FDR 2 -> 1.15, 3 -> 1.0, 4 -> 0.85, 5 -> 0.7
            fixture_factor = 1.0 + (3.0 - fdr) * 0.15

        # blend pieces – light but sensible
        # base_per90 ~ mix of season PPG & recent per90 & xGI
        base_per90 = (
            0.5 * ppg +
            0.35 * (form_per90 / 90.0) +
            0.15 * last5_xgi
        )

        xpts = base_per90 * minutes_factor * fixture_factor
        expected_points[pid] = xpts

        merged = row.to_dict()
        merged.update(
            {
                "last5_minutes": last5_mins,
                "last5_points": last5_pts,
                "last5_xgi": last5_xgi,
                "gw_xmins": gw_xmins,
                "expected_points_gw": xpts,
            }
        )
        players_info[pid] = merged

    return expected_points, players_info


# ---------- XI PICKER ----------


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
    by_pos: Dict[str, List[int]] = {"GK": [], "DEF": [], "MID": [], "FWD": []}

    for pid in squad_ids:
        info = player_lookup.get(pid)
        if not info:
            continue
        pos = info.get("position")
        if pos not in by_pos:
            continue
        by_pos[pos].append(pid)

    # sort each bucket by xPts desc
    for pos in by_pos:
        by_pos[pos].sort(key=lambda p: expected_points.get(p, 0.0), reverse=True)

    xi: List[int] = []

    # 1 GK
    if by_pos["GK"]:
        xi.append(by_pos["GK"][0])
    else:
        # no GK? Just take top 11, but this should never happen
        pool = [p for pos in by_pos for p in by_pos[pos]]
        pool.sort(key=lambda p: expected_points.get(p, 0.0), reverse=True)
        xi = pool[:11]
        total = sum(expected_points.get(pid, 0.0) for pid in xi)
        return xi, total

    # minimums: 3 DEF, 2 MID, 1 FWD
    for pos, min_needed in (("DEF", 3), ("MID", 2), ("FWD", 1)):
        xi.extend(by_pos[pos][:min_needed])

    pos_counts = {
        "GK": 1,
        "DEF": min(3, len(by_pos["DEF"])),
        "MID": min(2, len(by_pos["MID"])),
        "FWD": min(1, len(by_pos["FWD"])),
    }

    # remaining pool (DEF/MID/FWD beyond mins)
    remaining: List[int] = []
    for pos in ("DEF", "MID", "FWD"):
        remaining.extend(by_pos[pos][pos_counts[pos]:])

    remaining.sort(key=lambda p: expected_points.get(p, 0.0), reverse=True)

    max_limits = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}

    while len(xi) < 11 and remaining:
        pid = remaining.pop(0)
        pos = player_lookup[pid]["position"]
        if pos_counts.get(pos, 0) < max_limits[pos]:
            xi.append(pid)
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

    total = sum(expected_points.get(pid, 0.0) for pid in xi)
    return xi, total


# ---------- TRANSFER SEARCH ----------


def _build_team_counts(squad_ids: Set[int], player_lookup: Dict[int, Dict]) -> Dict[int, int]:
    """team_id -> count in current squad."""
    counts: Dict[int, int] = {}
    for pid in squad_ids:
        info = player_lookup.get(pid)
        if not info:
            continue
        team_id = _safe_int(info.get("team_id"), 0)
        if team_id:
            counts[team_id] = counts.get(team_id, 0) + 1
    return counts


def search_best_single_transfer(
    bears_squad: Dict,
    player_lookup: Dict[int, Dict],
    expected_points: Dict[int, float],
) -> Dict:
    """
    Scan all 1-transfer moves (same position, within budget, obeys 3-per-team)
    and return a dict describing the best one compared to HOLD.
    """
    picks = bears_squad["picks"]
    squad_ids: Set[int] = {p["element"] for p in picks}
    bank = bears_squad["bank"]

    base_xi, base_total = pick_best_xi(list(squad_ids), player_lookup, expected_points)

    best = {
        "type": "HOLD",
        "out": None,
        "in": None,
        "gain_vs_hold": 0.0,
        "new_xi": base_xi,
        "new_total": base_total,
    }

    # lookup helpers
    pos = {pid: player_lookup[pid]["position"] for pid in player_lookup}
    price = {pid: _price_millions(player_lookup[pid].get("now_cost")) for pid in player_lookup}
    team_id = {pid: _safe_int(player_lookup[pid].get("team_id"), 0) for pid in player_lookup}

    # only consider realistic transfer-in candidates: xMins >= 40
    candidates = [
        pid for pid in player_lookup
        if _safe_float(player_lookup[pid].get("gw_xmins", 0.0)) >= 40.0
    ]

    base_team_counts = _build_team_counts(squad_ids, player_lookup)

    for out_pid in squad_ids:
        out_pos = pos.get(out_pid)
        if not out_pos:
            continue

        budget = bank + price.get(out_pid, 0.0)
        out_team = team_id.get(out_pid, 0)

        for in_pid in candidates:
            if in_pid in squad_ids:
                continue
            if pos.get(in_pid) != out_pos:
                continue

            in_price = price.get(in_pid, 99.0)
            if in_price > budget:
                continue

            in_team = team_id.get(in_pid, 0)
            # 3-from-a-team rule
            new_team_counts = dict(base_team_counts)
            if out_team:
                new_team_counts[out_team] = new_team_counts.get(out_team, 0) - 1
            if in_team:
                new_team_counts[in_team] = new_team_counts.get(in_team, 0) + 1
                if new_team_counts[in_team] > 3:
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
    bears_squad: Dict,
    player_lookup: Dict[int, Dict],
    expected_points: Dict[int, float],
) -> Dict:
    """
    From the best 1FT result, try a second transfer for a -4 hit.
    Only keep it if (new_total - 4) > best_ft["new_total"].
    """
    picks = bears_squad["picks"]
    starting_ids: Set[int] = {p["element"] for p in picks}

    pos = {pid: player_lookup[pid]["position"] for pid in player_lookup}
    price = {pid: _price_millions(player_lookup[pid].get("now_cost")) for pid in player_lookup}
    team_id = {pid: _safe_int(player_lookup[pid].get("team_id"), 0) for pid in player_lookup}

    # apply first transfer (if any)
    if best_ft["type"] == "FT":
        starting_ids.remove(best_ft["out"])
        starting_ids.add(best_ft["in"])

    bank = bears_squad["bank"]
    if best_ft["type"] == "FT":
        bank = bank + price.get(best_ft["out"], 0.0) - price.get(best_ft["in"], 0.0)

    base_xi, base_total = pick_best_xi(list(starting_ids), player_lookup, expected_points)

    best_hit = {
        "type": "NONE",
        "out": None,
        "in": None,
        "gain_vs_best_ft_minus4": 0.0,
        "new_xi": best_ft["new_xi"],
        "new_total": best_ft["new_total"],
    }

    base_team_counts = _build_team_counts(starting_ids, player_lookup)

    candidates = [
        pid for pid in player_lookup
        if _safe_float(player_lookup[pid].get("gw_xmins", 0.0)) >= 40.0
    ]

    for out_pid in starting_ids:
        out_pos = pos.get(out_pid)
        if not out_pos:
            continue

        budget = bank + price.get(out_pid, 0.0)
        out_team = team_id.get(out_pid, 0)

        for in_pid in candidates:
            if in_pid in starting_ids:
                continue
            if pos.get(in_pid) != out_pos:
                continue

            in_price = price.get(in_pid, 99.0)
            if in_price > budget:
                continue

            in_team = team_id.get(in_pid, 0)

            new_team_counts = dict(base_team_counts)
            if out_team:
                new_team_counts[out_team] = new_team_counts.get(out_team, 0) - 1
            if in_team:
                new_team_counts[in_team] = new_team_counts.get(in_team, 0) + 1
                if new_team_counts[in_team] > 3:
                    continue

            new_squad = list(starting_ids - {out_pid} | {in_pid})
            xi, total = pick_best_xi(new_squad, player_lookup, expected_points)

            # compare against best_ft total including -4
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
    print(f"➡️  Using GW{gw} for reco")

    bears_entry_path = ENTRIES_DIR / BEARS_ENTRY_PATTERN.format(gw=gw)
    wigan_entry_path = ENTRIES_DIR / WIGAN_ENTRY_PATTERN.format(gw=gw)

    bears_entry = load_squad(bears_entry_path)
    wigan_entry = load_squad(wigan_entry_path)

    bears_summary = summarise_xi(bears_entry)
    wigan_summary = summarise_xi(wigan_entry)

    player_lookup = build_player_lookup(feed_df)

    # global xPts for all players
    expected_points_all, players_info_all = compute_expected_points_all(
        feed_df, recent_form
    )

    # extract Bears players subset with merged info
    bears_ids = {p["element"] for p in bears_entry["picks"]}
    bears_players_info = {
        pid: players_info_all[pid]
        for pid in bears_ids
        if pid in players_info_all
    }

    best_ft = search_best_single_transfer(
        bears_entry, player_lookup, expected_points_all
    )
    best_hit = maybe_search_second_transfer_hit(
        best_ft, bears_entry, player_lookup, expected_points_all
    )

    out = {
        "gw": gw,
        "generated_utc": datetime.utcnow().isoformat(),
        "bears": bears_summary,
        "wigan": wigan_summary,
        "bears_players": bears_players_info,
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

    # write GW-specific + stable "latest" file
    gw_path = BASE_DIR / OUT_RECO_TEMPLATE.format(gw=gw)
    save_json(gw_path, out)
    save_json(OUT_RECO_LATEST, out)

    print("🎉 build_reco.py complete")


if __name__ == "__main__":
    main()
