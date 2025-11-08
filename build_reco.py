#!/usr/bin/env python3
"""
build_reco.py

Full Birmingham Bears optimiser scaffold vs Wigan Witches.

Inputs (from public/):
  - bootstrap.json
  - meta.json
  - fixtures.json
  - feed_players.csv
  - recent_form.json
  - entries/bears_gw{gw}.json
  - entries/wigan_gw{gw}.json

Outputs:
  - public/bears_reco_gw{gw}.json     (per-GW snapshot)
  - public/bears_reco_latest.json     (always latest snapshot)

JSON structure:
{
  "gw": 11,
  "generated_utc": "...",
  "bears": {...},
  "wigan": {...},
  "transfers": [
     { "label": "hold", ... },
     { "label": "best_ft", ... },
     { "label": "best_hit_minus4", ... }
  ],
  "notes": [...]
}
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


# ---------- BASIC UTILS ----------

def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
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


def detect_current_gw(meta: dict, bootstrap: dict) -> int:
    """
    Decide which GW we're working on, preferring meta.json if present.
    """
    for key in ("current_event", "event", "gw"):
        if key in meta and isinstance(meta[key], int):
            gw = meta[key]
            print(f"ℹ️ GW from meta.json: {gw}")
            return gw

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


# ---------- SQUAD SUMMARY ----------

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
        # NOTE: event_transfers is "transfers used", not true FTs.
        "free_transfers": entry_hist.get("event_transfers", 0),
        "chip_active": entry_data.get("active_chip"),
        "captain": captain,
        "vice_captain": vice,
        "starting_xi": starting,
        "bench": bench,
    }


def get_bank(entry_data: dict) -> float:
    hist = entry_data.get("entry_history", {}) or {}
    return hist.get("bank", 0) / 10.0


# ---------- PLAYER TABLE UTILS ----------

def build_player_lookup(feed_df: pd.DataFrame) -> Dict[int, Dict]:
    """
    Map player_id -> row dict from feed_players.csv, auto-detecting ID column.
    Handles 'player_id', 'id', 'element', 'code'.
    """
    id_col = None
    for candidate in ["player_id", "id", "element", "code"]:
        if candidate in feed_df.columns:
            id_col = candidate
            break

    if not id_col:
        raise KeyError(
            f"No valid player ID column in feed_players.csv. "
            f"Columns = {feed_df.columns.tolist()}"
        )

    lookup: Dict[int, Dict] = {}
    for _, row in feed_df.iterrows():
        pid = int(row[id_col])
        lookup[pid] = row.to_dict()
    return lookup


def detect_price_key(sample_row: dict) -> str:
    for cand in ["now_cost", "price", "value", "cost"]:
        if cand in sample_row:
            return cand
    raise KeyError(f"No price field found in sample row: keys={list(sample_row.keys())}")


def detect_team_key(sample_row: dict) -> str:
    for cand in ["team_id", "team", "teamid"]:
        if cand in sample_row:
            return cand
    raise KeyError(f"No team field found in sample row: keys={list(sample_row.keys())}")


def is_viable_candidate(info: dict) -> bool:
    """
    Guardrails for transfer-in pool:
      - Not clearly injured/suspended/ruled out.
      - If doubtful, needs decent chance of playing.
      - Needs reasonable projected minutes.
    """
    status = (info.get("status") or "").lower()  # a,d,i,s,u etc.
    chance = info.get("chance_play_next")
    chance_val = _safe_float(chance, 100.0)

    gw_xmins = _safe_float(info.get("gw_xmins", 0.0))

    # Hard exclusions
    if status in ("i", "u", "s"):  # injured, unavailable, suspended
        return False

    # Doubtful with very low chance
    if status == "d" and chance_val < 50:
        return False

    # Needs at least 40 mins projection
    if gw_xmins < 40:
        return False

    return True


# ---------- MULTI-GW FIXTURE SCORE ----------

def build_multi_gw_fixture_scores(fixtures: list, gw: int, horizon: int = 3) -> Dict[int, float]:
    """
    For a window of [gw, gw+horizon-1], compute a simple fixture score per team:
      - Uses FDR (team_h_difficulty/team_a_difficulty).
      - Converts FDR to "goodness" (6 - FDR).
      - Averages across the window.
    Returns: {team_id: avg_goodness}
    """
    window = set(range(gw, gw + horizon))
    total: Dict[int, float] = {}
    count: Dict[int, int] = {}

    for f in fixtures:
        ev = f.get("event")
        if ev not in window:
            continue

        team_h = f.get("team_h")
        team_a = f.get("team_a")
        fdr_h = _safe_int(f.get("team_h_difficulty"), 3)
        fdr_a = _safe_int(f.get("team_a_difficulty"), 3)

        # FDR 1 → 5 (great), FDR 5 → 1 (awful)
        score_h = 6 - fdr_h
        score_a = 6 - fdr_a

        for tid, s in ((team_h, score_h), (team_a, score_a)):
            if tid is None:
                continue
            total[tid] = total.get(tid, 0.0) + s
            count[tid] = count.get(tid, 0) + 1

    avg: Dict[int, float] = {}
    for tid, val in total.items():
        avg[tid] = val / max(count.get(tid, 1), 1)
    return avg


# ---------- EXPECTED POINTS MODEL ----------

def compute_expected_points_all(
    feed_df: pd.DataFrame,
    recent_form: dict,
    multi_fix_scores: Dict[int, float] | None,
    team_key: str,
) -> Dict[int, float]:
    """
    Compute xPts for ALL players in feed_players.csv using:
      - points_per_game
      - last5_points / last5_minutes (per 90)
      - last5_xgi
      - gw_xmins
      - fixture difficulty for this GW
      - multi-GW fixture goodness (this + next 2 GWs)
    Returns: player_id -> expected_points_gw (with multi-GW adjustment baked in).
    """
    player_lookup = build_player_lookup(feed_df)
    expected_points: Dict[int, float] = {}

    for pid, base in player_lookup.items():
        # recent_form.json: keyed by string player ID
        f = recent_form.get(str(pid), {}) if isinstance(recent_form, dict) else {}
        last5_pts = f.get("last5_points", 0)
        last5_mins = f.get("last5_minutes", 0) or 0
        last5_xgi = f.get("last5_xgi", 0.0)

        # per-90 form – fallback to PPG if no minutes
        if last5_mins > 0:
            form_per90 = last5_pts / (last5_mins / 90.0)
        else:
            form_per90 = _safe_float(base.get("points_per_game", 0.0))

        gw_xmins = _safe_float(base.get("gw_xmins", 80.0))
        minutes_factor = gw_xmins / 90.0

        fixture_fdr = base.get("gw_fdr")
        fixture_fdr = int(fixture_fdr) if fixture_fdr is not None else 3
        fixture_factor = 1.0 + (3 - fixture_fdr) * 0.15  # FDR 2 → >1, FDR 4 → <1

        ppg = _safe_float(base.get("points_per_game", 0.0))

        base_xpts = (0.6 * ppg + 0.4 * form_per90 + 0.1 * last5_xgi) * minutes_factor * fixture_factor

        # multi-GW fixture adjustment (teams with better next 2–3 GWs get a slight bump)
        team_id = base.get(team_key)
        multi_score = 3.0
        if team_id is not None and multi_fix_scores is not None:
            multi_score = multi_fix_scores.get(int(team_id), 3.0)
        # centre around 3, scale lightly
        multi_factor = 1.0 + 0.05 * (multi_score - 3.0)

        xpts = base_xpts * multi_factor
        expected_points[pid] = xpts

    return expected_points


# ---------- XI PICKER & SCORING ----------

def pick_best_xi(
    squad_ids: List[int],
    player_lookup: Dict[int, Dict],
    expected_points: Dict[int, float],
) -> List[int]:
    """
    Greedy XI selector:
      - exactly 1 GK
      - 3–5 DEF, 2–5 MID, 1–3 FWD
      - maximise sum of expected_points
    Returns only the XI player IDs; scoring is separate.
    """
    by_pos = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for pid in squad_ids:
        info = player_lookup.get(pid)
        if not info:
            continue
        pos = info.get("position")
        if pos not in by_pos:
            continue
        by_pos[pos].append(pid)

    # sort each group by xPts desc
    for pos in by_pos:
        by_pos[pos].sort(key=lambda p: expected_points.get(p, 0.0), reverse=True)

    xi: List[int] = []

    # 1 GK
    if by_pos["GK"]:
        xi.append(by_pos["GK"][0])

    # minimums
    minimums = [("DEF", 3), ("MID", 2), ("FWD", 1)]
    pos_counts = {"GK": 1 if by_pos["GK"] else 0, "DEF": 0, "MID": 0, "FWD": 0}

    for pos, min_needed in minimums:
        picks = by_pos[pos][:min_needed]
        xi.extend(picks)
        pos_counts[pos] = len(picks)

    # remaining players pool (DEF/MID/FWD beyond minimums)
    remaining: List[int] = []
    for pos in ("DEF", "MID", "FWD"):
        remaining.extend(by_pos[pos][pos_counts[pos]:])

    remaining.sort(key=lambda p: expected_points.get(p, 0.0), reverse=True)

    max_limits = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}

    while len(xi) < 11 and remaining:
        pid = remaining.pop(0)
        info = player_lookup.get(pid, {})
        pos = info.get("position")
        if not pos or pos not in max_limits:
            continue
        if pos_counts[pos] < max_limits[pos]:
            xi.append(pid)
            pos_counts[pos] += 1

    return xi


def score_xi(
    xi: List[int],
    expected_points: Dict[int, float],
    wigan_ids: set[int],
    diff_bonus: float = 0.05,
    shared_penalty: float = 0.02,
) -> Tuple[float, float]:
    """
    Score an XI in two ways:
      - raw_total: sum of xPts
      - effective_score: raw_total adjusted for differentials vs Wigan

    differential logic:
      - if player NOT in wigan_ids → +diff_bonus * xPts
      - if player IS in wigan_ids → -shared_penalty * xPts
    """
    raw_total = 0.0
    bonus = 0.0

    for pid in xi:
        xp = expected_points.get(pid, 0.0)
        raw_total += xp
        if pid in wigan_ids:
            bonus -= shared_penalty * xp
        else:
            bonus += diff_bonus * xp

    effective_score = raw_total + bonus
    return effective_score, raw_total


# ---------- FPL RULES HELPERS ----------

def respects_team_limit(
    squad_ids: List[int],
    player_lookup: Dict[int, Dict],
    team_key: str,
) -> bool:
    """
    Check FPL's 'max 3 players per club' rule.
    """
    counts: Dict[int, int] = {}
    for pid in squad_ids:
        info = player_lookup.get(pid, {})
        team_id = info.get(team_key)
        if team_id is None:
            continue
        team_id = int(team_id)
        counts[team_id] = counts.get(team_id, 0) + 1
        if counts[team_id] > 3:
            return False
    return True


# ---------- TRANSFER SEARCH ----------

def search_best_single_transfer(
    bears_entry: dict,
    player_lookup: Dict[int, Dict],
    expected_points: Dict[int, float],
    wigan_ids: set[int],
    team_key: str,
) -> Dict[str, Any]:
    """
    Scan all 1-transfer moves (same position, within budget, respects team limit)
    and return the best one vs hold, using differential-aware effective score.
    """
    picks = bears_entry.get("picks", []) or []
    squad_ids = {p["element"] for p in picks}
    bank = get_bank(bears_entry)

    if not squad_ids:
        return {
            "type": "HOLD",
            "out": None,
            "in": None,
            "gain_vs_hold": 0.0,
            "new_xi": [],
            "new_total_raw": 0.0,
            "base_xi": [],
            "base_total_raw": 0.0,
        }

    sample_row = next(iter(player_lookup.values()))
    price_key = detect_price_key(sample_row)

    pos_map = {pid: player_lookup[pid].get("position") for pid in squad_ids if pid in player_lookup}
    price = {pid: _safe_float(info.get(price_key)) for pid, info in player_lookup.items()}

    # baseline XI and scores
    base_xi = pick_best_xi(list(squad_ids), player_lookup, expected_points)
    base_eff, base_raw = score_xi(base_xi, expected_points, wigan_ids)

    best = {
        "type": "HOLD",
        "out": None,
        "in": None,
        "gain_vs_hold": 0.0,
        "new_xi": base_xi,
        "new_total_raw": base_raw,
        "base_xi": base_xi,
        "base_total_raw": base_raw,
    }

    # candidate pool: viable + decent minutes
    candidates = [
        pid for pid, info in player_lookup.items()
        if is_viable_candidate(info)
    ]

    for out_pid in squad_ids:
        out_pos = pos_map.get(out_pid)
        if not out_pos:
            continue

        budget = bank + price.get(out_pid, 0.0)

        for in_pid in candidates:
            if in_pid in squad_ids:
                continue

            info_in = player_lookup.get(in_pid, {})
            if info_in.get("position") != out_pos:
                continue
            if price.get(in_pid, 1e9) > budget:
                continue

            new_squad = list((squad_ids - {out_pid}) | {in_pid})
            if not respects_team_limit(new_squad, player_lookup, team_key):
                continue

            xi = pick_best_xi(new_squad, player_lookup, expected_points)
            eff_score, raw_score = score_xi(xi, expected_points, wigan_ids)
            gain = eff_score - base_eff

            if gain > best["gain_vs_hold"]:
                best.update(
                    {
                        "type": "FT",
                        "out": out_pid,
                        "in": in_pid,
                        "gain_vs_hold": gain,
                        "new_xi": xi,
                        "new_total_raw": raw_score,
                    }
                )

    return best


def maybe_search_second_transfer_hit(
    best_ft: Dict[str, Any],
    bears_entry: dict,
    player_lookup: Dict[int, Dict],
    expected_points: Dict[int, float],
    wigan_ids: set[int],
    team_key: str,
) -> Dict[str, Any]:
    """
    From the best 1FT result, try a second transfer for a -4 hit.
    Only keep it if (effective_score_new - 4) actually beats best_ft's effective score.
    """
    picks = bears_entry.get("picks", []) or []
    base_squad_ids = {p["element"] for p in picks}

    sample_row = next(iter(player_lookup.values()))
    price_key = detect_price_key(sample_row)

    price = {pid: _safe_float(info.get(price_key)) for pid, info in player_lookup.items()}

    if best_ft["type"] == "FT":
        base_squad_ids.remove(best_ft["out"])
        base_squad_ids.add(best_ft["in"])

    # adjust "bank" after best_ft
    bank = get_bank(bears_entry)
    if best_ft["type"] == "FT":
        bank = bank + price.get(best_ft["out"], 0.0) - price.get(best_ft["in"], 0.0)

    # recompute baseline for this partially updated squad
    base_xi = pick_best_xi(list(base_squad_ids), player_lookup, expected_points)
    base_eff, base_raw = score_xi(base_xi, expected_points, wigan_ids)

    # effective score of best FT path (for comparison)
    ft_eff_score = base_eff
    if best_ft["type"] == "FT":
        # if FT exists we use its XI for effective baseline
        ft_eff_score, _ = score_xi(best_ft["new_xi"], expected_points, wigan_ids)

    best_hit = {
        "type": "NONE",
        "out": None,
        "in": None,
        "gain_vs_best_ft_minus4": 0.0,
        "new_xi": best_ft["new_xi"],
        "new_total_raw": best_ft["new_total_raw"],
    }

    candidates = [
        pid for pid, info in player_lookup.items()
        if is_viable_candidate(info)
    ]
    pos_map = {pid: player_lookup[pid].get("position") for pid in base_squad_ids if pid in player_lookup}

    for out_pid in base_squad_ids:
        out_pos = pos_map.get(out_pid)
        if not out_pos:
            continue

        budget = bank + price.get(out_pid, 0.0)

        for in_pid in candidates:
            if in_pid in base_squad_ids:
                continue
            if player_lookup[in_pid].get("position") != out_pos:
                continue
            if price.get(in_pid, 1e9) > budget:
                continue

            new_squad = list((base_squad_ids - {out_pid}) | {in_pid})
            if not respects_team_limit(new_squad, player_lookup, team_key):
                continue

            xi = pick_best_xi(new_squad, player_lookup, expected_points)
            eff_score, raw_score = score_xi(xi, expected_points, wigan_ids)

            # compare against FT effective score with -4 hit
            effective_gain = (eff_score - 4.0) - ft_eff_score

            if effective_gain > best_hit["gain_vs_best_ft_minus4"]:
                best_hit.update(
                    {
                        "type": "HIT-4",
                        "out": out_pid,
                        "in": in_pid,
                        "gain_vs_best_ft_minus4": effective_gain,
                        "new_xi": xi,
                        "new_total_raw": raw_score,
                    }
                )

    if best_hit["type"] == "NONE" or best_hit["gain_vs_best_ft_minus4"] <= 0:
        return {
            "type": "NONE",
            "out": None,
            "in": None,
            "gain_vs_best_ft_minus4": 0.0,
            "new_xi": best_ft["new_xi"],
            "new_total_raw": best_ft["new_total_raw"],
        }

    return best_hit


# ---------- MAIN ----------

def main() -> None:
    print("🔄 build_reco.py starting …")

    meta = load_json(META_PATH) if META_PATH.exists() else {}
    bootstrap = load_json(BOOTSTRAP_PATH)
    fixtures = load_json(FIXTURES_PATH) if FIXTURES_PATH.exists() else []
    feed_df = pd.read_csv(FEED_PLAYERS_PATH)
    recent_form = load_json(RECENT_FORM_PATH) if RECENT_FORM_PATH.exists() else {}

    gw = detect_current_gw(meta, bootstrap)
    print(f"➡️  Using GW{gw} for reco")

    bears_entry_path = ENTRIES_DIR / BEARS_ENTRY_PATTERN.format(gw=gw)
    wigan_entry_path = ENTRIES_DIR / WIGAN_ENTRY_PATTERN.format(gw=gw)

    if not bears_entry_path.exists():
        raise FileNotFoundError(f"Bears entry file missing: {bears_entry_path}")
    if not wigan_entry_path.exists():
        raise FileNotFoundError(f"Wigan entry file missing: {wigan_entry_path}")

    bears_entry = load_json(bears_entry_path)
    wigan_entry = load_json(wigan_entry_path)

    bears_summary = summarise_xi(bears_entry)
    wigan_summary = summarise_xi(wigan_entry)

    # Wigan squad for differential logic
    wigan_picks = wigan_entry.get("picks", []) or []
    wigan_ids = {p["element"] for p in wigan_picks}

    # Player lookup & multi-GW fixture scores
    player_lookup = build_player_lookup(feed_df)
    sample_row = next(iter(player_lookup.values()))
    team_key = detect_team_key(sample_row)

    multi_fix_scores = None
    if fixtures:
        multi_fix_scores = build_multi_gw_fixture_scores(fixtures, gw, horizon=3)

    # Expected points (with multi-GW baked in)
    expected_points = compute_expected_points_all(
        feed_df, recent_form, multi_fix_scores, team_key
    )

    # Transfer search
    best_ft = search_best_single_transfer(
        bears_entry, player_lookup, expected_points, wigan_ids, team_key
    )
    best_hit = maybe_search_second_transfer_hit(
        best_ft, bears_entry, player_lookup, expected_points, wigan_ids, team_key
    )

    # Baseline from best_ft structure (raw XI points)
    base_xi = best_ft.get("base_xi", [])
    base_raw = best_ft.get("base_total_raw", 0.0)

    transfers_block = [
        {
            "label": "hold",
            "type": "HOLD",
            "hits": 0,
            "out": None,
            "in": None,
            "gain_vs_hold": 0.0,
            "expected_points_new_xi": base_raw,
            "xi": base_xi,
        },
        {
            "label": "best_ft",
            "type": best_ft["type"],
            "hits": 0,
            "out": best_ft["out"],
            "in": best_ft["in"],
            "gain_vs_hold": best_ft["gain_vs_hold"],
            "expected_points_new_xi": best_ft["new_total_raw"],
            "xi": best_ft["new_xi"],
        },
        {
            "label": "best_hit_minus4",
            "type": best_hit["type"],
            "hits": 4 if best_hit["type"] == "HIT-4" else 0,
            "out": best_hit["out"],
            "in": best_hit["in"],
            "gain_vs_hold_after_hits": best_hit["gain_vs_best_ft_minus4"],
            "expected_points_new_xi": best_hit["new_total_raw"],
            "xi": best_hit["new_xi"],
        },
    ]

    out = {
        "gw": gw,
        "generated_utc": datetime.utcnow().isoformat(),
        "bears": bears_summary,
        "wigan": wigan_summary,
        "transfers": transfers_block,
        "notes": [
            "Model includes: xPts from PPG + last-5 form + xGI + xMins + fixture difficulty.",
            "Multi-GW fixture run (this + next ~2 GWs) is baked into xPts.",
            "Transfers are rival-aware vs Wigan and use guardrails for injuries/minutes.",
        ],
    }

    # Write per-GW and latest files
    gw_filename = OUT_RECO_GW_PATTERN.format(gw=gw)
    out_path_gw = PUBLIC_DIR / gw_filename
    out_path_latest = PUBLIC_DIR / OUT_RECO_LATEST

    save_json(out_path_gw, out)
    save_json(out_path_latest, out)

    print(f"🎉 build_reco.py complete for GW{gw}")


if __name__ == "__main__":
    main()
