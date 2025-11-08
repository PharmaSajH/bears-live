#!/usr/bin/env python3
"""
build_reco.py (auto-column-safe)

Produces ChatGPT recommendation files:
    public/bears_reco_gw{gw}.json
    public/bears_reco_latest.json
"""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple
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

# ---------- UTILS ----------

def load_json(path: Path):
    with path.open() as f:
        return json.load(f)

def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ wrote {path.relative_to(BASE_DIR)}")

def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default

def _safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default

# ---------- GW DETECTION ----------
def detect_current_gw(meta: dict, bootstrap: dict) -> int:
    for key in ("current_event", "event", "gw"):
        if key in meta and isinstance(meta[key], int):
            print(f"GW from meta.json: {meta[key]}")
            return meta[key]
    for e in bootstrap.get("events", []):
        if e.get("is_current"):
            return e["id"]
    finished = [e for e in bootstrap.get("events", []) if e.get("finished")]
    if finished:
        return max(finished, key=lambda x: x["id"])["id"]
    raise RuntimeError("Could not determine GW")

# ---------- PLAYER LOOKUP ----------
def build_player_lookup(feed_df: pd.DataFrame) -> Dict[int, Dict]:
    """Auto-detect player_id column name"""
    col_map = [c for c in feed_df.columns if c.lower() in ("player_id", "id", "element", "code")]
    if not col_map:
        raise KeyError("feed_players.csv missing any player id column (expected player_id/id/element/code)")
    pid_col = col_map[0]
    lookup = {}
    for _, row in feed_df.iterrows():
        pid = int(row[pid_col])
        lookup[pid] = row.to_dict()
    return lookup

def _price_millions(raw):
    v = _safe_float(raw)
    return v / 10 if v > 20 else v

# ---------- EXPECTED POINTS ----------
def compute_expected_points_all(feed_df, recent_form):
    expected, info = {}, {}
    pid_col = [c for c in feed_df.columns if c.lower() in ("player_id", "id", "element", "code")][0]
    for _, row in feed_df.iterrows():
        pid = int(row[pid_col])
        rec = recent_form.get(str(pid), {}) if isinstance(recent_form, dict) else {}
        mins = _safe_float(rec.get("last5_minutes", 0))
        pts = _safe_float(rec.get("last5_points", 0))
        xgi = _safe_float(rec.get("last5_xgi", 0))
        form90 = pts * 90 / mins if mins > 0 else 0
        ppg = _safe_float(row.get("points_per_game", 0))
        gw_xmins = _safe_float(row.get("gw_xmins", 80))
        fdr = _safe_float(row.get("gw_fdr", 3))
        fix_factor = 1 + (3 - fdr) * 0.15
        xpts = (0.5*ppg + 0.35*(form90/90) + 0.15*xgi) * (gw_xmins/90) * fix_factor
        expected[pid] = xpts
        d = row.to_dict()
        d.update({"expected_points_gw": xpts})
        info[pid] = d
    return expected, info

# ---------- PICK XI ----------
def pick_best_xi(squad, lookup, expected):
    by_pos = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for pid in squad:
        pos = lookup.get(pid, {}).get("position")
        if pos in by_pos:
            by_pos[pos].append(pid)
    for p in by_pos:
        by_pos[p].sort(key=lambda x: expected.get(x, 0), reverse=True)
    xi = []
    if by_pos["GK"]: xi.append(by_pos["GK"][0])
    xi += by_pos["DEF"][:3] + by_pos["MID"][:2] + by_pos["FWD"][:1]
    remain = []
    for p in ("DEF","MID","FWD"):
        remain += by_pos[p][len(xi):]
    remain.sort(key=lambda x: expected.get(x,0), reverse=True)
    while len(xi)<11 and remain:
        xi.append(remain.pop(0))
    return xi, sum(expected.get(x,0) for x in xi)

# ---------- TRANSFER SEARCH ----------
def _build_team_counts(squad, lookup):
    counts = {}
    for pid in squad:
        t = _safe_int(lookup.get(pid, {}).get("team_id"))
        counts[t] = counts.get(t, 0) + 1
    return counts

def search_best_single_transfer(squad, lookup, expected):
    picks = {p["element"] for p in squad["picks"]}
    bank = squad["bank"]
    base_xi, base_total = pick_best_xi(list(picks), lookup, expected)
    best = {"type":"HOLD","out":None,"in":None,"gain_vs_hold":0,"new_xi":base_xi,"new_total":base_total}
    pos = {p: lookup[p].get("position") for p in lookup}
    price = {p: _price_millions(lookup[p].get("now_cost")) for p in lookup}
    team = {p: _safe_int(lookup[p].get("team_id")) for p in lookup}
    base_team = _build_team_counts(picks, lookup)
    for out_pid in picks:
        budget = bank + price.get(out_pid,0)
        out_team = team.get(out_pid)
        for in_pid in lookup:
            if in_pid in picks: continue
            if pos.get(in_pid)!=pos.get(out_pid): continue
            if price.get(in_pid,99)>budget: continue
            in_team = team.get(in_pid)
            new_team = dict(base_team)
            new_team[out_team] = new_team.get(out_team,0)-1
            new_team[in_team] = new_team.get(in_team,0)+1
            if new_team[in_team]>3: continue
            xi, total = pick_best_xi(list(picks-{out_pid}|{in_pid}), lookup, expected)
            gain = total-base_total
            if gain>best["gain_vs_hold"]:
                best={"type":"FT","out":out_pid,"in":in_pid,"gain_vs_hold":gain,"new_xi":xi,"new_total":total,"base_xi":base_xi,"base_total":base_total}
    best["base_xi"]=base_xi
    best["base_total"]=base_total
    return best

def maybe_search_second_transfer_hit(best_ft,squad,lookup,expected):
    picks={p["element"] for p in squad["picks"]}
    if best_ft["type"]=="FT": picks.remove(best_ft["out"]); picks.add(best_ft["in"])
    bank=squad["bank"]
    price={p:_price_millions(lookup[p].get("now_cost")) for p in lookup}
    pos={p:lookup[p].get("position") for p in lookup}
    team={p:_safe_int(lookup[p].get("team_id")) for p in lookup}
    if best_ft["type"]=="FT": bank+=price.get(best_ft["out"],0)-price.get(best_ft["in"],0)
    base_team=_build_team_counts(picks,lookup)
    best_hit={"type":"NONE","out":None,"in":None,"gain_vs_best_ft_minus4":0,"new_xi":best_ft["new_xi"],"new_total":best_ft["new_total"]}
    for out_pid in picks:
        budget=bank+price.get(out_pid,0)
        out_team=team.get(out_pid)
        for in_pid in lookup:
            if in_pid in picks: continue
            if pos.get(in_pid)!=pos.get(out_pid): continue
            if price.get(in_pid,99)>budget: continue
            in_team=team.get(in_pid)
            new_team=dict(base_team)
            new_team[out_team]=new_team.get(out_team,0)-1
            new_team[in_team]=new_team.get(in_team,0)+1
            if new_team[in_team]>3: continue
            xi,total=pick_best_xi(list(picks-{out_pid}|{in_pid}),lookup,expected)
            eff_gain=(total-4)-best_ft["new_total"]
            if eff_gain>best_hit["gain_vs_best_ft_minus4"]:
                best_hit={"type":"HIT-4","out":out_pid,"in":in_pid,"gain_vs_best_ft_minus4":eff_gain,"new_xi":xi,"new_total":total}
    if best_hit["gain_vs_best_ft_minus4"]<=0: return {"type":"NONE","out":None,"in":None,"gain_vs_best_ft_minus4":0,"new_xi":best_ft["new_xi"],"new_total":best_ft["new_total"]}
    return best_hit

# ---------- MAIN ----------
def main():
    print("🔄 build_reco.py starting …")
    meta=load_json(META_PATH); bootstrap=load_json(BOOTSTRAP_PATH)
    feed_df=pd.read_csv(FEED_PLAYERS_PATH); recent_form=load_json(RECENT_FORM_PATH)
    gw=detect_current_gw(meta,bootstrap); print(f"Using GW{gw}")
    bears=load_json(ENTRIES_DIR/BEARS_ENTRY_PATTERN.format(gw=gw))
    wigan=load_json(ENTRIES_DIR/WIGAN_ENTRY_PATTERN.format(gw=gw))
    player_lookup=build_player_lookup(feed_df)
    expected,info=compute_expected_points_all(feed_df,recent_form)
    best_ft=search_best_single_transfer(bears,player_lookup,expected)
    best_hit=maybe_search_second_transfer_hit(best_ft,bears,player_lookup,expected)
    out={"gw":gw,"generated_utc":datetime.utcnow().isoformat(),
         "bears":bears,"wigan":wigan,
         "model":{"baseline":{"xi":best_ft["base_xi"],"expected_points":best_ft["base_total"]},
                  "best_ft":best_ft,"best_hit_minus4":best_hit}}
    save_json(BASE_DIR/OUT_RECO_TEMPLATE.format(gw=gw),out)
    save_json(OUT_RECO_LATEST,out)
    print("🎉 build_reco.py complete")

if __name__=="__main__":
    main()
