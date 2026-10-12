# -*- coding: utf-8 -*-
"""
P0 -- Paper number verification (recomputes every manuscript number from
authoritative raw files; see Unified_Specification_v2.md, sec. 8 "recompute,
do not trust archived statistics"). Emits P0_paper_numbers.json.

Authoritative inputs (canonical table in Unified_Specification_v2.md):
  cities21_locked_s2 -> 04_实验数据/数值结果/21城市精确MILP验证结果_两阶段版_withS2.json
  enum_full_s2       -> poa_full_per_config_s2.json
  subsidy (classic)  -> 04_实验数据/数值结果/21城市_补贴后.json
Channel-decomposition / layered bounds are taken from the derived CSVs
(Table6_2_layered_stats.csv, 21城市通道分解_stats.json) which the audit already
validated; they are reported here as cross-checked references.
"""
import json, os, statistics

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
NUM = os.path.join(ROOT, "results", "main")
C21 = os.path.join(NUM, "21城市精确MILP验证结果_两阶段版_withS2.json")
ENUM = os.path.join(NUM, "results_enumeration.json")
SUBS = os.path.join(NUM, "21城市_补贴后.json")
OUT = {}

# --------------------------------------------------------------------------
# Pre-flight: this test recomputes the published constants from the archived
# experiment outputs.  Those outputs are intentionally not redistributed
# (see data/external/DATA_SOURCES.md, section 2.2), so in a fresh checkout the
# test skips with instructions instead of raising FileNotFoundError.
# --------------------------------------------------------------------------
_MISSING = [p for p in (C21, ENUM, SUBS) if not os.path.exists(p)]
if _MISSING:
    raise SystemExit(
        "SKIP verify_paper_numbers: the following archived outputs are absent:\n"
        + "\n".join("  - " + os.path.relpath(p, ROOT) for p in _MISSING)
        + "\n\nRegenerate them first (see docs/REPRODUCIBILITY.md):\n"
          "    python code/03_experiments/exp1_21cities.py\n"
          "    python code/03_experiments/exp2_enumeration.py --workers 8\n"
          "    python code/03_experiments/exp3_subsidy.py")

def pct(x): return round(100.0 * x, 4)
def med(lst): return round(statistics.median(lst), 4)

# --------------------------------------------------------------------------
# 1. 21 real cities (locked specification)
# --------------------------------------------------------------------------
d = json.load(open(C21, encoding="utf-8"))
assert d["nash_mode"] == "locked"
assert abs(d["budget_per_station_wan"] - 36.0) < 1e-9
cities = list(d["results"].values())
poa = [c["PoA"] for c in cities]
OUT["c21_n"] = len(cities)
OUT["c21_mean_PoA"] = round(statistics.mean(poa), 4)
OUT["c21_max_PoA"] = round(max(poa), 4)
OUT["c21_min_PoA"] = round(min(poa), 4)

# Unified gap = milp_F2 - nash_F2. Because F2 is stored to 4 decimals, the
# smallest representable gap is 1e-4. Xi'an sits exactly on that floor
# (-1e-4) and is reported as a TIE (Unified_Spec v2 sec.4.2).
TIEFLOOR = 1e-4
milp_win = tie = nash_win = 0
cov_gap = 0
cov_gaps = []
pos_dF2 = []
for c in cities:
    g = round(c["milp_F2"] - c["nash_F2"], 4)
    if g > TIEFLOOR:
        milp_win += 1
    elif g < -TIEFLOOR:
        nash_win += 1
    else:
        tie += 1
    cg = abs(c["milp_F1"] - c["nash_F1"])
    if cg > 1e-9:
        cov_gap += 1
        cov_gaps.append(cg)
OUT["c21_milp_strict"] = milp_win
OUT["c21_tie"] = tie
OUT["c21_nash_strict"] = nash_win
OUT["c21_cov_gap_count"] = cov_gap
OUT["c21_cov_gap_range"] = [round(min(cov_gaps), 4), round(max(cov_gaps), 4)]

# channel SHARES (share of dF2) over the 18 cities with |dF2| > 0
# (definition of 21城vs枚举锁定层对照报告.md, matching 21城市通道分解_stats.json)
nonzero_dF2 = [c for c in cities if abs(c["milp_F2"] - c["nash_F2"]) > 0]
sh1 = [c["ch_S1"] / (c["milp_F2"] - c["nash_F2"]) for c in nonzero_dF2]
sh2 = [c["ch_S2"] / (c["milp_F2"] - c["nash_F2"]) for c in nonzero_dF2]
shc = [c["ch_cov"] / (c["milp_F2"] - c["nash_F2"]) for c in nonzero_dF2]
OUT["c21_channel_median_S1"] = med(sh1)
OUT["c21_channel_median_S2"] = med(sh2)
OUT["c21_channel_median_cov"] = med(shc)
OUT["c21_identity_resid_max"] = round(max(c["identity_resid"] for c in cities), 12)

# --------------------------------------------------------------------------
# 2. Full enumeration (classic specification)
# --------------------------------------------------------------------------
de = json.load(open(ENUM, encoding="utf-8"))["results"]
def is_degenerate(c): return c.get("milp_F2", 0) == 0 and c.get("nash_F2", 0) == 0
valid = [c for c in de if not is_degenerate(c)]
OUT["enum_raw_count"] = len(de)
OUT["enum_valid_count"] = len(valid)
epoa = [c["PoA"] for c in valid]
OUT["enum_mean_PoA"] = round(statistics.mean(epoa), 4)
OUT["enum_median_PoA"] = round(statistics.median(epoa), 4)
OUT["enum_max_PoA"] = round(max(epoa), 4)
OUT["enum_min_PoA"] = round(min(epoa), 4)
m = tie_e = n_e = 0
for c in valid:
    g = c["milp_F2"] - c["nash_F2"]
    if g > 0: m += 1
    elif g < 0: n_e += 1
    else: tie_e += 1
OUT["enum_pct_MILP_strict"] = pct(m / len(valid))
OUT["enum_pct_equal"] = pct(tie_e / len(valid))
OUT["enum_pct_Nash_strict"] = pct(n_e / len(valid))

# --------------------------------------------------------------------------
# 3. Subsidy experiment (classic specification)
# --------------------------------------------------------------------------
ds = json.load(open(SUBS, encoding="utf-8"))
sres = ds["results"] if isinstance(ds, dict) else ds
subs = list(sres.values()) if isinstance(sres, dict) else sres
imp = [c["nash_F2_improve_pct"] for c in subs if "nash_F2_improve_pct" in c]
base = [c["PoA_base"] for c in subs if "PoA_base" in c]
cons = [c["PoA_sub_consistent"] for c in subs if "PoA_sub_consistent" in c]
OUT["subsidy_n"] = len(subs)
OUT["subsidy_mean_improve_pct"] = round(statistics.mean(imp), 4)
OUT["subsidy_mean_PoA_base"] = round(statistics.mean(base), 4)
OUT["subsidy_mean_PoA_subsidized"] = round(statistics.mean(cons), 4)

json.dump(OUT, open(os.path.join(NUM, "03_实验代码", "调试辅助工具", "P0_paper_numbers.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(json.dumps(OUT, indent=2, ensure_ascii=False))

EXP = {
    "c21_mean_PoA": 1.0267, "c21_max_PoA": 1.0663, "c21_min_PoA": 0.9903,
    "c21_milp_strict": 15, "c21_tie": 4, "c21_nash_strict": 2,
    "c21_cov_gap_count": 13,
    "c21_channel_median_S1": 0.2164, "c21_channel_median_S2": 0.4928, "c21_channel_median_cov": 0.2792,
    "enum_valid_count": 15207, "enum_mean_PoA": 1.0398, "enum_max_PoA": 1.4398,
    "enum_pct_MILP_strict": 87.84, "enum_pct_equal": 3.42, "enum_pct_Nash_strict": 8.74,
    "subsidy_mean_improve_pct": 5.97, "subsidy_mean_PoA_base": 1.0394,
    "subsidy_mean_PoA_subsidized": 1.0497,
}
print("\n=== CROSS-CHECK vs Unified_Specification_v2.md ===")
ok = True
for k, v in EXP.items():
    got = OUT.get(k)
    if got is None:
        print(f"  [MISS] {k}: not computed"); continue
    if isinstance(v, float) and abs(got - v) <= max(0.005, abs(v) * 0.005):
        print(f"  [OK ] {k}: {got} ~ {v}")
    elif isinstance(v, int) and got == v:
        print(f"  [OK ] {k}: {got} == {v}")
    else:
        ok = False
        print(f"  [DIFF] {k}: got {got} expected {v}")
print("ALL OK" if ok else "REVIEW NEEDED")
