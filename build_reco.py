#!/usr/bin/env python3
"""
build_reco.py

Reads live data files from public/ and public/entries/ and produces:

  - public/bears_reco_gw{gw}.json
  - public/bears_reco_latest.json

Now also auto-ingests:
  - public/model_state.json  (from post_gw_learning.py)
  - public/chip_plan.json    (from chip_plan.py)

So ChatGPT only needs ONE file (bears_reco_latest.json) to see:
  - Bears + Wigan squads (XI, bench, bank, value, captain/vice)
  - Model xPts for Bears squad
  - Best FT and optional -4 path
  - Current learning knobs (ceiling, minutes risk, hit thresholds)
  - Chip planning hints
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
META_PATH = PUBLIC_DIR / "meta.json"
FEED_PLAYERS_PATH = PUBLIC_DIR / "feed_players.csv"
RECENT_FORM_PATH = PUBLIC_DIR / "recent_form.json"
MODEL_STATE_PATH = PUBLIC_DIR / "model_state.json"
CHIP_PLAN_PATH = PUBLIC_DIR / "chip_plan.json"

BEARS_ENTRY_PATTERN = "bears_gw{gw}.json"
WIGAN_ENTRY_PATTERN = "wigan_gw{gw}.json"

OUT_RECO_TEMPLATE = PUBLIC_DIR / "bears_reco_gw{gw}.json"
OUT_RECO_LATEST = PUBLIC_DIR / "bears_reco_latest.json"


# ---------- BASIC UTILS ----------

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


def _price_millions(raw) -> float:
    """Convert FPL now_cost style (e.g. 59) to 5.9, else pass through."""
    v = _safe_float(raw)
    return v / 10.0 if v > 20 else v


# ---------- GW DETECTION ----------

def detect_current_gw(meta: dict, bootstrap: dict) -> int:
    # prefer meta.json
    for key in ("current_event", "event", "gw"):
        if key in meta and isinstance(meta[key], int):
            print(f"ℹ️ GW from meta.json: {meta[key]}")
            return meta[key]

    events = bootstrap.get("events") or []
    current = next((e for e in events if e.get("is_current")), None)
    if current:
        print(f"ℹ️ GW from bootstrap (is_current): {current['id']}")
        return current["id"]

    finished = [e for e in events if e.get("finished")]
    if finished:
        gw = max(finished, key=lambda e: e["id"])["id"]
        print(f"ℹ️ GW from last finished event: {gw}")
        return gw

    raise RuntimeError("Could not determine current gameweek")


# ---------- ENTRY → SQUAD STRUCTURES ----------

def load_entry_as_squad(entry_path: Path) -> Dict:
    """
    Read bears_gw{gw}.json / wigan_gw{gw}.json and normalise to:
      {
        "entry_id": int,
        "bank": float (millions),
        "team_value": float (millions),
        "chip_active": str|None,
        "picks": [ {...}, ... ]
      }
    """
    data = load_json(entry_path, default={})
    hist = data.get("entry_history") or {}
    bank = hist.get("bank", 0) / 10.0
    value = hist.get("value", 0) / 10.0

    return {
        "entry_id": data.get("entry"),
        "bank": bank,
        "team_value": value,
        "chip_active": data.get("active_chip"),
        "picks": data.get("picks", []),
    }


def summarise_xi(squad: Dict) -> Dict:
    """
    From the squad dict above, extract captain, XI, bench lists.
    """
    picks = squad.get("picks", [])
    starting = [p["element"] for p in picks if p.get("multiplier", 0) > 0]
    bench = [p["element"] for p in picks if p.get("multiplier", 0) == 0]

    captain = next((p["element"] for p in picks if p.get("is_captain")), None)
    vice = next((p["element"] for p in picks if p.get("is_vice_captain")), None)

    return {
        "entry_id": squad["entry_id"],
        "bank": squad["bank"],
        "team_value": squad["team_value"],
        "chip_active": squad["chip_active"],
        "captain": captain,
        "vice_captain": vice,
        "starting_xi": starting,
        "bench": bench,
    }


# ---------- PLAYER LOOKUP & XPTS ----------

def build_player_lookup(feed_df: pd.DataFrame) -> Dict[int, Dict]:
    """
    Map player_id -> row dict from feed_players.csv.

    Auto-detects the id column name (player_id / id / element / code).
    """
    candidates = [c for c in feed_df.columns
                  if c.lower() in ("player_id", "id", "element", "code")]
    if not candidates:
        raise KeyError(
            "feed_players.csv missing player id column "
            "(expected one of: player_id, id, element, code)"
        )
    pid_col = candidates[0]

    lookup: Dict[int, Dict] = {}
    for _, row in feed_df.iterrows():
        pid = int(row[pid_col])
        lookup[pid] = row.to_dict()
    return lookup


def default_model_state() -> dict:
    now = datetime.utcnow().isoformat()
    return {
        "version": 1,
        "created_utc": now,
        "last_updated_utc": now,
        "last_gw": None,
        "ceiling_aggression": 1.00,
        "minutes_risk_penalty": 1.00,
        "differential_weight": 1.00,
        "hit_threshold_ft_gain": 4.0,
        "hit_threshold_double_gain": 7.0,
        "history": [],
    }


def compute_expected_points_all(
    feed_df: pd.DataFrame,
    recent_form: Dict,
    model_state: Dict,
) -> Tuple[Dict[int, float], Dict[int, Dict]]:
    """
    Compute expected points for *all* players in feed_players.csv.
    Uses model_state knobs (ceiling_aggression, minutes_risk_penalty).
    """
    candidates = [c for c in feed_df.columns
                  if c.lower() in ("player_id", "id", "element", "code")]
    pid_col = candidates[0]

    ceil = float(model_state.get("ceiling_aggression", 1.0))
    mins_pen = float(model_state.get("minutes_risk_penalty", 1.0))

    expected: Dict[int, float] = {}
    info: Dict[int, Dict] = {}

    for _, row in feed_df.iterrows():
        pid = int(row[pid_col])

        rf = recent_form.get(str(pid), {}) if isinstance(recent_form, dict) else {}
        mins5 = _safe_float(rf.get("last5_minutes", 0))
        pts5 = _safe_float(rf.get("last5_points", 0))
        xgi5 = _safe_float(rf.get("last5_xgi", 0.0))

        form_per90 = (pts5 / mins5 * 90.0) if mins5 > 0 else 0.0

        ppg = _safe_float(row.get("points_per_game", 0.0))
        gw_xmins = _safe_float(row.get("gw_xmins", 80.0))
        fdr = _safe_float(row.get("gw_fdr", 3.0))

        base_minutes_factor = gw_xmins / 90.0

        # minutes risk penalty: if low xMins, scale down more
        if gw_xmins < 60:
            minutes_factor = base_minutes_factor / max(mins_pen, 0.1)
        else:
            minutes_factor = base_minutes_factor

        fixture_factor = 1.0 + (3.0 - fdr) * 0.15  # easier fixtures upweight, harder downweight

        base_xpts = (
            0.5 * ppg +
            0.35 * (form_per90 / 90.0) +
            0.15 * xgi5
        ) * minutes_factor * fixture_factor

        # ceiling aggression just scales the upside slightly
        xpts = base_xpts * ceil

        expected[pid] = xpts

        merged = row.to_dict()
        merged.update({
            "last5_minutes": mins5,
            "last5_points": pts5,
            "last5_xgi": xgi5,
            "gw_xmins": gw_xmins,
            "expected_points_gw": xpts,
        })
        info[pid] = merged

    return expected, info


# ---------- XI PICKER ----------

def pick_best_xi(
    squad_ids: List[int],
    player_lookup: Dict[int, Dict],
    expected_points: Dict[int, float],
) -> Tuple[List[int], float]:
    """
    Simple greedy optimiser that respects:

      - 1 GK
      - 3–5 DEF
      - 2–5 MID
      - 1–3 FWD
    """
    by_pos = {"GK": [], "DEF": [], "MID": [], "FWD": []}

    for pid in squad_ids:
        pos = player_lookup.get(pid, {}).get("position")
        if pos in by_pos:
            by_pos[pos].append(pid)

    for pos in by_pos:
        by_pos[pos].sort(key=lambda x: expected_points.get(x, 0.0), reverse=True)

    xi: List[int] = []

    # 1 GK
    if by_pos["GK"]:
        xi.append(by_pos["GK"][0])

    # minimum: 3 DEF, 2 MID, 1 FWD
    xi.extend(by_pos["DEF"][:3])
    xi.extend(by_pos["MID"][:2])
    xi.extend(by_pos["FWD"][:1])

    # counts
    pos_counts = {
        "GK": len([p for p in xi if player_lookup.get(p, {}).get("position") == "GK"]),
        "DEF": len([p for p in xi if player_lookup.get(p, {}).get("position") == "DEF"]),
        "MID": len([p for p in xi if player_lookup.get(p, {}).get("position") == "MID"]),
        "FWD": len([p for p in xi if player_lookup.get(p, {}).get("position") == "FWD"]),
    }

    max_limits = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}

    remaining: List[int] = []
    for pos in ("DEF", "MID", "FWD"):
        used = pos_counts[pos]
        remaining.extend(by_pos[pos][used:])

    remaining.sort(key=lambda x: expected_points.get(x, 0.0), reverse=True)

    while len(xi) < 11 and remaining:
        pid = remaining.pop(0)
        pos = player_lookup.get(pid, {}).get("position")
        if not pos:
            continue
        if pos_counts[pos] >= max_limits[pos]:
            continue
        xi.append(pid)
        pos_counts[pos] += 1

    total = sum(expected_points.get(pid, 0.0) for pid in xi)
    return xi, total


# ---------- TRANSFER SEARCH HELPERS ----------

def _team_counts(squad_ids: List[int], player_lookup: Dict[int, Dict]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for pid in squad_ids:
        tid = _safe_int(player_lookup.get(pid, {}).get("team_id"))
        counts[tid] = counts.get(tid, 0) + 1
    return counts


def search_best_single_transfer(
    squad: Dict,
    player_lookup: Dict[int, Dict],
    expected_points: Dict[int, float],
    hit_threshold_ft_gain: float,
) -> Dict:
    """
    Best 1FT compared to just holding.
    Only surfaces a "FT" if gain_vs_hold >= hit_threshold_ft_gain.
    """
    picks = {p["element"] for p in squad.get("picks", [])}
    bank = squad.get("bank", 0.0)

    base_xi, base_total = pick_best_xi(list(picks), player_lookup, expected_points)

    best = {
        "type": "HOLD",
        "out": None,
        "in": None,
        "gain_vs_hold": 0.0,
        "new_xi": base_xi,
        "new_total": base_total,
        "base_xi": base_xi,
        "base_total": base_total,
    }

    pos = {pid: player_lookup.get(pid, {}).get("position") for pid in player_lookup}
    price = {pid: _price_millions(player_lookup.get(pid, {}).get("now_cost")) for pid in player_lookup}
    team = {pid: _safe_int(player_lookup.get(pid, {}).get("team_id")) for pid in player_lookup}

    base_team_counts = _team_counts(list(picks), player_lookup)

    for out_pid in picks:
        out_pos = pos.get(out_pid)
        if not out_pos:
            continue

        budget = bank + price.get(out_pid, 0.0)
        out_team = team.get(out_pid)

        for in_pid in player_lookup.keys():
            if in_pid in picks:
                continue
            if pos.get(in_pid) != out_pos:
                continue
            if price.get(in_pid, 99.0) > budget:
                continue

            in_team = team.get(in_pid)
            new_team = dict(base_team_counts)
            new_team[out_team] = new_team.get(out_team, 0) - 1
            new_team[in_team] = new_team.get(in_team, 0) + 1
            if new_team[in_team] > 3:
                continue

            new_squad_ids = list((picks - {out_pid}) | {in_pid})
            xi, total = pick_best_xi(new_squad_ids, player_lookup, expected_points)
            gain = total - base_total

            if gain > best["gain_vs_hold"]:
                best.update(
                    {
                        "type": "FT",
                        "out": out_pid,
                        "in": in_pid,
                        "gain_vs_hold": gain,
                        "new_xi": xi,
                        "new_total": total,
                    }
                )

    # apply threshold: if the best gain is below threshold, treat as HOLD
    if best["type"] == "FT" and best["gain_vs_hold"] < hit_threshold_ft_gain:
        best["type"] = "HOLD"
        best["out"] = None
        best["in"] = None
        best["gain_vs_hold"] = 0.0
        best["new_xi"] = best["base_xi"]
        best["new_total"] = best["base_total"]

    return best


def maybe_search_second_transfer_hit(
    best_ft: Dict,
    squad: Dict,
    player_lookup: Dict[int, Dict],
    expected_points: Dict[int, float],
    hit_threshold_double_gain: float,
) -> Dict:
    """
    From best 1FT position, try a second transfer with a -4 hit.
    Only keep if:
        (new_total - 4) - best_ft["new_total"] >= hit_threshold_double_gain
    """
    picks = {p["element"] for p in squad.get("picks", [])}

    if best_ft.get("type") == "FT":
        picks.remove(best_ft["out"])
        picks.add(best_ft["in"])

    bank = squad.get("bank", 0.0)
    price = {pid: _price_millions(player_lookup.get(pid, {}).get("now_cost")) for pid in player_lookup}
    pos = {pid: player_lookup.get(pid, {}).get("position") for pid in player_lookup}
    team = {pid: _safe_int(player_lookup.get(pid, {}).get("team_id")) for pid in player_lookup}

    if best_ft.get("type") == "FT":
        bank = bank + price.get(best_ft["out"], 0.0) - price.get(best_ft["in"], 0.0)

    base_team_counts = _team_counts(list(picks), player_lookup)

    best_hit = {
        "type": "NONE",
        "out": None,
        "in": None,
        "gain_vs_best_ft_minus4": 0.0,
        "new_xi": best_ft["new_xi"],
        "new_total": best_ft["new_total"],
    }

    for out_pid in picks:
        out_pos = pos.get(out_pid)
        if not out_pos:
            continue

        budget = bank + price.get(out_pid, 0.0)
        out_team = team.get(out_pid)

        for in_pid in player_lookup.keys():
            if in_pid in picks:
                continue
            if pos.get(in_pid) != out_pos:
                continue
            if price.get(in_pid, 99.0) > budget:
                continue

            in_team = team.get(in_pid)
            new_team = dict(base_team_counts)
            new_team[out_team] = new_team.get(out_team, 0) - 1
            new_team[in_team] = new_team.get(in_team, 0) + 1
            if new_team[in_team] > 3:
                continue

            new_squad_ids = list((picks - {out_pid}) | {in_pid})
            xi, total = pick_best_xi(new_squad_ids, player_lookup, expected_points)

            effective_gain = (total - 4.0) - best_ft["new_total"]
            if effective_gain > best_hit["gain_vs_best_ft_minus4"]:
                best_hit.update(
                    {
                        "type": "HIT-4",
                        "out": out_pid,
                        "in": in_pid,
                        "gain_vs_best_ft_minus4": effective_gain,
                        "new_xi": xi,
                        "new_total": total,
                    }
                )

    # apply threshold
    if best_hit["type"] == "HIT-4" and best_hit["gain_vs_best_ft_minus4"] < hit_threshold_double_gain:
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

    meta = load_json(META_PATH, default={})
    bootstrap = load_json(BOOTSTRAP_PATH, default={})
    feed_df = pd.read_csv(FEED_PLAYERS_PATH)
    recent_form = load_json(RECENT_FORM_PATH, default={})

    # new: model_state + chip_plan (optional but auto-ingested)
    model_state = load_json(MODEL_STATE_PATH, default=default_model_state())
    if "version" not in model_state:
        model_state = default_model_state()

    chip_plan = load_json(CHIP_PLAN_PATH, default={})

    gw = detect_current_gw(meta, bootstrap)
    print(f"➡️ Using GW{gw} for reco")

    bears_entry_path = ENTRIES_DIR / BEARS_ENTRY_PATTERN.format(gw=gw)
    wigan_entry_path = ENTRIES_DIR / WIGAN_ENTRY_PATTERN.format(gw=gw)

    bears_squad = load_entry_as_squad(bears_entry_path)
    wigan_squad = load_entry_as_squad(wigan_entry_path)

    bears_summary = summarise_xi(bears_squad)
    wigan_summary = summarise_xi(wigan_squad)

    player_lookup = build_player_lookup(feed_df)

    expected_points, all_players_info = compute_expected_points_all(
        feed_df, recent_form, model_state
    )

    # hit thresholds from model_state
    hit_ft = float(model_state.get("hit_threshold_ft_gain", 4.0))
    hit_double = float(model_state.get("hit_threshold_double_gain", 7.0))

    best_ft = search_best_single_transfer(
        bears_squad, player_lookup, expected_points, hit_ft
    )
    best_hit = maybe_search_second_transfer_hit(
        best_ft, bears_squad, player_lookup, expected_points, hit_double
    )

    # only include Bears players (15-man squad) in the reco to keep size small
    bears_ids = {p["element"] for p in bears_squad.get("picks", [])}
    bears_players = {
        str(pid): all_players_info.get(pid, {}) for pid in bears_ids
    }

    out = {
        "gw": gw,
        "generated_utc": datetime.utcnow().isoformat(),
        "bears": bears_summary,
        "wigan": wigan_summary,
        "bears_players": bears_players,
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
        "learning": {
            "ceiling_aggression": model_state.get("ceiling_aggression"),
            "minutes_risk_penalty": model_state.get("minutes_risk_penalty"),
            "differential_weight": model_state.get("differential_weight"),
            "hit_threshold_ft_gain": model_state.get("hit_threshold_ft_gain"),
            "hit_threshold_double_gain": model_state.get("hit_threshold_double_gain"),
            "last_gw_learned": model_state.get("last_gw"),
        },
        "chips": chip_plan,   # full chip_plan.json embedded so I can see it from one file
    }

    # Write GW-specific + latest files
    gw_path = Path(str(OUT_RECO_TEMPLATE).format(gw=gw))
    save_json(gw_path, out)
    save_json(OUT_RECO_LATEST, out)

    print("🎉 build_reco.py complete")


if __name__ == "__main__":
    main()
