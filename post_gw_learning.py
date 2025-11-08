#!/usr/bin/env python3
"""
post_gw_learning.py

Lightweight post–Gameweek calibration for the Bears model.

Reads:
  - public/bears_history.json
  - public/wigan_history.json
  - public/bears_reco_gw{gw}.json (if present) or bears_reco_latest.json
  - public/model_state.json (optional, will be created if missing)

Writes:
  - public/model_state.json

The idea is:
  - Compare predicted vs actual Bears points.
  - Compare result vs Wigan.
  - Tweak:
      * ceiling_aggression
      * minutes_risk_penalty
      * differential_weight
      * hit_threshold_ft_gain
      * hit_threshold_double_gain
  - Append a short history record per GW.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"

BEARS_HISTORY_PATH = PUBLIC_DIR / "bears_history.json"
WIGAN_HISTORY_PATH = PUBLIC_DIR / "wigan_history.json"
MODEL_STATE_PATH = PUBLIC_DIR / "model_state.json"

RECO_LATEST_PATH = PUBLIC_DIR / "bears_reco_latest.json"
RECO_GW_PATTERN = "bears_reco_gw{gw}.json"


# ----------------- small helpers -----------------


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


def detect_latest_gw_from_history(hist: dict) -> int | None:
    """Return the last GW that has a 'current' record."""
    current = hist.get("current") or []
    if not current:
        return None
    latest = max(current, key=lambda r: r.get("event", 0))
    return latest.get("event")


def get_points_for_gw(hist: dict, gw: int) -> dict:
    """
    Return a tiny dict with Bears' GW points + hit cost etc.

    Structure of `current` list (FPL API) is like:
      {'event': 11, 'points': 62, 'event_transfers_cost': 4, ...}
    """
    current = hist.get("current") or []
    for rec in current:
        if rec.get("event") == gw:
            return {
                "points": rec.get("points", 0),
                "event_transfers": rec.get("event_transfers", 0),
                "event_transfers_cost": rec.get("event_transfers_cost", 0),
                "rank": rec.get("overall_rank"),
            }
    return {
        "points": 0,
        "event_transfers": 0,
        "event_transfers_cost": 0,
        "rank": None,
    }


def default_model_state() -> dict:
    now = datetime.utcnow().isoformat()
    return {
        "version": 1,
        "created_utc": now,
        "last_updated_utc": now,
        "last_gw": None,
        # tunable knobs
        "ceiling_aggression": 1.00,        # >1 = more upside chasing
        "minutes_risk_penalty": 1.00,      # >1 = more cautious on risky minutes
        "differential_weight": 1.00,       # >1 = more focus on non-template picks
        "hit_threshold_ft_gain": 4.0,      # how much xPts a FT must gain vs hold
        "hit_threshold_double_gain": 7.0,  # how much for a 2-move hit
        "history": [],                     # list of per-GW summary dicts
    }


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ----------------- main calibration logic -----------------


def main():
    print("🔄 post_gw_learning.py starting …")

    bears_hist = load_json(BEARS_HISTORY_PATH)
    wigan_hist = load_json(WIGAN_HISTORY_PATH)

    latest_gw = detect_latest_gw_from_history(bears_hist)
    if latest_gw is None:
        print("⚠️ No GW history found for Bears; nothing to learn from.")
        return

    print(f"ℹ️ Latest finished GW in history: {latest_gw}")

    # Load existing state or create defaults
    state = load_json(MODEL_STATE_PATH, default=default_model_state())
    if "version" not in state:
        state = default_model_state()

    if state.get("last_gw") == latest_gw:
        # Already processed this GW – just refresh timestamp and exit
        state["last_updated_utc"] = datetime.utcnow().isoformat()
        save_json(MODEL_STATE_PATH, state)
        print(f"ℹ️ GW{latest_gw} already learned; nothing to do.")
        return

    # --- actual vs predicted for this GW ---
    bears_points = get_points_for_gw(bears_hist, latest_gw)
    wigan_points = get_points_for_gw(wigan_hist, latest_gw)

    bears_score = bears_points["points"]
    wigan_score = wigan_points["points"]
    margin_vs_wigan = bears_score - wigan_score
    hit_cost = bears_points["event_transfers_cost"] or 0

    print(f"ℹ️ Bears GW{latest_gw} points: {bears_score} (hit cost: {hit_cost})")
    print(f"ℹ️ Wigan GW{latest_gw} points: {wigan_score} (margin: {margin_vs_wigan})")

    # Try to load the GW-specific reco first; if missing, fall back to latest
    reco_path = PUBLIC_DIR / RECO_GW_PATTERN.format(gw=latest_gw)
    reco = load_json(reco_path, default=None)
    if not reco or reco.get("gw") != latest_gw:
        print(f"⚠️ {reco_path.name} missing or GW mismatch – falling back to bears_reco_latest.json")
        reco = load_json(RECO_LATEST_PATH, default={})

    baseline_expected = (
        reco.get("model", {})
        .get("baseline", {})
        .get("expected_points", None)
    )

    if baseline_expected is None:
        print("⚠️ No baseline expected points found in reco; using simple deltas only.")
        prediction_error = None
    else:
        prediction_error = bears_score - baseline_expected
        print(f"ℹ️ Baseline expected xPts: {baseline_expected:.2f}")
        print(f"ℹ️ Prediction error (actual - expected): {prediction_error:.2f}")

    # --------- update knobs based on what happened ---------
    ceil = float(state.get("ceiling_aggression", 1.0))
    mins_pen = float(state.get("minutes_risk_penalty", 1.0))
    diff_w = float(state.get("differential_weight", 1.0))
    hit_ft = float(state.get("hit_threshold_ft_gain", 4.0))
    hit_double = float(state.get("hit_threshold_double_gain", 7.0))

    # 1) prediction error -> ceiling vs safety
    if prediction_error is not None:
        if prediction_error < -10:  # badly under-performed
            ceil *= 0.95
            mins_pen *= 1.05
            print("↘️ Big under-performance: nudging towards safety (less ceiling, more mins penalty).")
        elif prediction_error > 10:  # crushed expectations
            ceil *= 1.05
            mins_pen *= 0.97
            print("↗️ Strong over-performance: nudging towards more ceiling / a bit less mins penalty.")

    # 2) rivalry result vs Wigan -> differential weighting
    if margin_vs_wigan < 0:
        diff_w *= 1.05
        print("⚔️ Lost to Wigan: increasing differential_weight slightly.")
    elif margin_vs_wigan > 0:
        diff_w *= 0.98
        print("🏆 Beat Wigan: easing differential_weight slightly.")

    # 3) hit evaluation – was paying for hits worth it?
    if hit_cost > 0 and baseline_expected is not None:
        # crude: if actual points are worse than expected by more than the hit cost, treat as bad hit
        if bears_score + hit_cost < baseline_expected:
            hit_ft += 0.25
            hit_double += 0.5
            print("💸 Hit looked poor vs model – raising hit thresholds.")
        else:
            hit_ft -= 0.25
            hit_double -= 0.5
            print("💰 Hit looked acceptable – lowering hit thresholds slightly.")

    # clamp to sane ranges
    ceil = clip(ceil, 0.6, 1.6)
    mins_pen = clip(mins_pen, 0.6, 1.6)
    diff_w = clip(diff_w, 0.6, 1.6)
    hit_ft = clip(hit_ft, 3.0, 6.0)
    hit_double = clip(hit_double, 6.0, 10.0)

    # write back
    state["ceiling_aggression"] = round(ceil, 3)
    state["minutes_risk_penalty"] = round(mins_pen, 3)
    state["differential_weight"] = round(diff_w, 3)
    state["hit_threshold_ft_gain"] = round(hit_ft, 2)
    state["hit_threshold_double_gain"] = round(hit_double, 2)
    state["last_gw"] = latest_gw
    state["last_updated_utc"] = datetime.utcnow().isoformat()

    # append history record (keep last ~30)
    history = state.get("history") or []
    history.append(
        {
            "gw": latest_gw,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "bears_points": bears_score,
            "wigan_points": wigan_score,
            "margin_vs_wigan": margin_vs_wigan,
            "hit_cost": hit_cost,
            "baseline_expected": baseline_expected,
            "prediction_error": prediction_error,
            "knobs": {
                "ceiling_aggression": state["ceiling_aggression"],
                "minutes_risk_penalty": state["minutes_risk_penalty"],
                "differential_weight": state["differential_weight"],
                "hit_threshold_ft_gain": state["hit_threshold_ft_gain"],
                "hit_threshold_double_gain": state["hit_threshold_double_gain"],
            },
        }
    )
    # keep last 30 to avoid unbounded growth
    state["history"] = history[-30:]

    save_json(MODEL_STATE_PATH, state)
    print("🎉 post_gw_learning.py complete")


if __name__ == "__main__":
    main()
