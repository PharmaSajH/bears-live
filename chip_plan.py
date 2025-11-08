#!/usr/bin/env python3
"""
chip_plan.py

Very lightweight, fully automated chip-planning helper.

Reads:
  - public/bootstrap.json
  - public/fixtures.json
  - public/bears_history.json
  - public/model_state.json (optional)

Writes:
  - public/chip_plan.json
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
    current = next((e for e in events if e.get("is_current")), None)
    if current:
        return current.get("id")
    unfinished = [e for e in events if not e.get("finished")]
    if unfinished:
        return min(unfinished, key=lambda e: e["id"])["id"]
    if events:
        return max(events, key=lambda e: e["id"])["id"]
    return None


def detect_last_gw_from_history(hist: dict) -> int | None:
    current = hist.get("current") or []
    if not current:
        return None
    latest = max(current, key=lambda r: r.get("event", 0))
    return latest.get("event")


def chips_available(hist: dict) -> dict:
    used_raw = {c.get("name") for c in (hist.get("chips") or [])}

    def is_unused(names):
        return all(n not in used_raw for n in names)

    return {
        "wildcard": is_unused(["wildcard", "wildcard2"]),
        "freehit": is_unused(["freehit"]),
        "benchboost": is_unused(["bboost", "benchboost"]),
        "triple_captain": is_unused(["3xc", "triple_captain"]),
    }


def build_fixture_difficulty_by_gw(fixtures: list) -> dict[int, dict]:
    gw_stats = defaultdict(lambda: {"easy": 0, "hard": 0, "total": 0})
    for f in fixtures:
        gw = f.get("event")
        if gw is None:
            continue
        h = f.get("team_h_difficulty")
        a = f.get("team_a_difficulty")
        if h is not None:
            gw_stats[gw]["total"] += 1
            if h <= 2:
                gw_stats[gw]["easy"] += 1
            if h >= 4:
                gw_stats[gw]["hard"] += 1
        if a is not None:
            gw_stats[gw]["total"] += 1
            if a <= 2:
                gw_stats[gw]["easy"] += 1
            if a >= 4:
                gw_stats[gw]["hard"] += 1
    return gw_stats


def suggest_chips(current_gw, max_gw, gw_stats, available, model_state):
    upper = min(max_gw, current_gw + 16)
    future_gws = [gw for gw in sorted(gw_stats) if current_gw <= gw <= upper]
    if not future_gws:
        return {"suggested": {}, "rationale": {"note": "No future GWs found in fixtures.json"}}

    best_easy_gw = None
    worst_easy_gw = None
    for gw in future_gws:
        stats = gw_stats[gw]
        if best_easy_gw is None or stats["easy"] > gw_stats[best_easy_gw]["easy"]:
            best_easy_gw = gw
        if worst_easy_gw is None or stats["easy"] < gw_stats[worst_easy_gw]["easy"]:
            worst_easy_gw = gw

    suggestions, rationale = {}, {}
    ceil = float(model_state.get("ceiling_aggression", 1.0))

    # Triple Captain
    if available.get("triple_captain"):
        tc_gw = best_easy_gw if ceil >= 1.1 else best_easy_gw + 1
        suggestions["triple_captain"] = tc_gw
        rationale["triple_captain"] = (
            f"GW{tc_gw}: {gw_stats[tc_gw]['easy']} easy fixtures. "
            f"ceiling_aggression={ceil:.2f}"
        )

    # Bench Boost
    if available.get("benchboost"):
        bb_gw = suggestions.get("triple_captain", best_easy_gw) + 1
        suggestions["benchboost"] = bb_gw
        rationale["benchboost"] = (
            f"GW{bb_gw}: {gw_stats.get(bb_gw, {}).get('easy', 0)} easy fixtures. "
            "Bench Boost follows TC week."
        )

    # Free Hit
    if available.get("freehit"):
        fh_gw = worst_easy_gw
        suggestions["freehit"] = fh_gw
        rationale["freehit"] = (
            f"GW{fh_gw}: {gw_stats[fh_gw]['easy']} easy fixtures. "
            "Free Hit helps navigate toughest GW."
        )

    # Wildcard
    if available.get("wildcard"):
        wc_gw = max(current_gw + 1, suggestions.get("freehit", worst_easy_gw) - 3)
        suggestions["wildcard"] = wc_gw
        rationale["wildcard"] = f"Wildcard ~3 weeks before FH (target GW{wc_gw})."

    return {"suggested": suggestions, "rationale": rationale}


def main():
    print("🔄 chip_plan.py starting …")

    bootstrap = load_json(BOOTSTRAP_PATH)
    fixtures = load_json(FIXTURES_PATH, default=[])
    bears_hist = load_json(BEARS_HISTORY_PATH)
    model_state = load_json(MODEL_STATE_PATH, default={})

    current_from_bootstrap = detect_current_gw_from_bootstrap(bootstrap)
    current_from_hist = detect_last_gw_from_history(bears_hist)
    current_gw = current_from_hist or current_from_bootstrap

    if current_gw is None:
        print("⚠️ Could not determine current GW; aborting chip plan.")
        return

    max_gw = max((e.get("id", 0) for e in (bootstrap.get("events") or [])), default=current_gw)
    available = chips_available(bears_hist)
    gw_stats = build_fixture_difficulty_by_gw(fixtures)

    plan_core = suggest_chips(current_gw, max_gw, gw_stats, available, model_state)

    out = {
        "generated_utc": datetime.utcnow().isoformat(),
        "current_gw": current_gw,
        "max_gw": max_gw,
        "chips_available": available,
        "model_state_snapshot": model_state,
        "plan": plan_core,
    }

    save_json(CHIP_PLAN_PATH, out)
    print("🎉 chip_plan.py complete")


if __name__ == "__main__":
    main()
