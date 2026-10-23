# -*- coding: utf-8 -*-
"""Recompute every number the manuscript quotes, from the exact-run outputs.

This is the package's regression guard.  It reads the *exact* artefacts only
(single-MILP Phase 2, HiGHS) and re-derives each published statistic with plain
NumPy/SciPy/Pandas, then compares it with the value printed in the manuscript.

Inputs (all shipped in a full checkout):
    results/main/results_enumeration_exact.json          full enumeration, n=15,207
    results/main/21cities_exact_locked.json              21 cities, coverage-locked
    results/main/21cities_exact_classic.json             21 cities, classic (contrast)
    results/main/_subsidy_exact/subsidy_exact.json       V2 subsidy, classic baseline
    results/sensitivity/cap_floor_theta_*_exact_per_config.json   theta sweep

Output:
    results/tables/P0_paper_numbers.json   (recomputed values)
    exit code 1 if any check fails

Run:  python tests/verify_paper_numbers.py
"""
import json
import os
import statistics
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
NUM = os.path.join(ROOT, "results", "main")
SENS = os.path.join(ROOT, "results", "sensitivity")
OUT_PATH = os.path.join(ROOT, "results", "tables", "P0_paper_numbers.json")

ENUM_EXACT = os.path.join(NUM, "results_enumeration_exact.json")
C21_LOCKED = os.path.join(NUM, "21cities_exact_locked.json")
C21_CLASSIC = os.path.join(NUM, "21cities_exact_classic.json")
SUBSIDY = os.path.join(NUM, "_subsidy_exact", "subsidy_exact.json")
THETA_FILES = {t: os.path.join(SENS, "cap_floor_theta_%s_exact_per_config.json" % t)
               for t in ("0.50", "0.60", "0.70")}

W1, W2, W3, S3 = 0.2, 0.3, 0.5, 0.60
TOL_TIE = 1e-12          # PoA noise floor used for "exact tie"
TIEFLOOR_21 = 1e-4       # 21-city gate: F2 is stored to 4 decimals

REQUIRED = [ENUM_EXACT, C21_LOCKED, C21_CLASSIC, SUBSIDY] + list(THETA_FILES.values())
_MISSING = [p for p in REQUIRED if not os.path.exists(p)]
if _MISSING:
    # This is a SKIP, not a failure: the exact outputs under results/main/ are
    # derived from the restricted competition rasters and are therefore NOT
    # redistributed (see .gitignore and docs/DATA_SOURCES.md).  They exist only
    # on a machine that has regenerated them.  Exit code 2 distinguishes "could
    # not verify" from "verification failed" (exit code 1).
    print("SKIP verify_paper_numbers: %d required exact output(s) absent:" % len(_MISSING))
    for p in _MISSING:
        print("  - " + os.path.relpath(p, ROOT))
    print(
        "\nThese files derive from restricted inputs and are intentionally not\n"
        "shipped. Regenerate them first (about 1 h on 8 cores):\n"
        "    python code/03_experiments/exp2_enumeration_exact.py\n"
        "    python code/04_sensitivity/cap_floor_sensitivity.py --exact\n"
        "    python code/03_experiments/exp1_21cities.py --exact-phase2 --nash-mode locked"
        " --budget-per-station 36 --output results/main/21cities_exact_locked.json\n"
        "    python code/03_experiments/exp1_21cities.py --exact-phase2 --nash-mode classic"
        " --budget-per-station 36 --output results/main/21cities_exact_classic.json\n"
        "    python code/04_sensitivity/cluster_radius_21.py          # optional extra cells\n"
        "    python code/03_experiments/exp3_subsidy.py --exact-phase2\n")
    sys.exit(2)

OUT = {}
FAILS = []


def check(name, got, want, tol=0.0005, fmt="%.4f"):
    """Record one manuscript number and compare it with the recomputed value."""
    OUT[name] = float(got) if isinstance(got, float) else int(got)
    if isinstance(want, int):
        ok = int(got) == int(want)
    else:
        ok = abs(float(got) - float(want)) <= tol
    if not ok:
        FAILS.append(name)
    shown = fmt % float(got) if isinstance(got, float) else str(got)
    print("  [%s] %-42s got %-12s manuscript %s"
          % ("OK " if ok else "DIFF", name, shown, want))


def load(p):
    return json.load(open(p, encoding="utf-8"))


# ===========================================================================
# 1. 21 real cities, coverage-locked (tab:poa21, Sec. 5)
# ===========================================================================
print("\n=== 1. 21 cities, coverage-locked ===")
d21 = load(C21_LOCKED)
assert d21["nash_mode"] == "locked" and abs(d21["budget_per_station_wan"] - 36.0) < 1e-9
c21 = d21["results"]
poa21 = np.array([v["PoA"] for v in c21.values()])
gap21 = np.array([v["milp_F2"] - v["nash_F2"] for v in c21.values()])

check("c21.n", len(c21), 21)
check("c21.mean_PoA", float(poa21.mean()), 1.0376, 5e-5)
check("c21.median_PoA", float(np.median(poa21)), 1.0357, 5e-5)
check("c21.max_PoA", float(poa21.max()), 1.0791, 5e-5)
check("c21.min_PoA", float(poa21.min()), 1.0000, 5e-5)
check("c21.milp_strict", int((gap21 > TIEFLOOR_21).sum()), 18)
check("c21.ties", int((np.abs(gap21) <= TIEFLOOR_21).sum()), 3)
check("c21.nash_strict", int((gap21 < -TIEFLOOR_21).sum()), 0)
check("c21.smallest_nonzero_gap",
      float(np.abs(gap21[np.abs(gap21) > TIEFLOOR_21]).min()), 4.0e-4, 1e-9, "%.1e")

dcov21 = np.array([v["milp_cov"] - v["nash_cov"] for v in c21.values()])
lag = np.abs(dcov21)
check("c21.coverage_lag_count", int((lag > 1e-9).sum()), 13)
check("c21.coverage_lag_min", float(lag[lag > 1e-9].min()), 0.0029, 5e-5)
check("c21.coverage_lag_max", float(lag.max()), 0.0190, 5e-5)

nz = [v for v in c21.values() if abs(v["milp_F2"] - v["nash_F2"]) > 0]
sh = np.array([[v["ch_S1"] / (v["milp_F2"] - v["nash_F2"]),
                v["ch_S2"] / (v["milp_F2"] - v["nash_F2"]),
                v["ch_cov"] / (v["milp_F2"] - v["nash_F2"])] for v in nz])
check("c21.n_nonzero_gap", len(nz), 18)
check("c21.channel_median_S1", float(np.median(sh[:, 0])), 0.055, 5e-4)
check("c21.channel_median_S2", float(np.median(sh[:, 1])), 0.704, 5e-4)
check("c21.channel_median_cov", float(np.median(sh[:, 2])), 0.204, 5e-4)
check("c21.channel_mean_S1", float(sh[:, 0].mean()), 0.024, 5e-4)
check("c21.channel_mean_S2", float(sh[:, 1].mean()), 0.786, 5e-4)
check("c21.channel_mean_cov", float(sh[:, 2].mean()), 0.190, 5e-4)
check("c21.identity_resid_max", float(max(v["identity_resid"] for v in c21.values())),
      0.0, 1e-9, "%.1e")

aligned = [k for k, v in c21.items() if abs(v["milp_cov"] - v["nash_cov"]) <= 1e-12]
check("c21.coverage_aligned", len(aligned), 8)
al_nz = [v for k, v in c21.items() if k in aligned and abs(v["milp_F2"] - v["nash_F2"]) > 0]
check("c21.aligned_nonzero_gap", len(al_nz), 5)
if al_nz:
    med_s2 = statistics.median([v["ch_S2"] / (v["milp_F2"] - v["nash_F2"]) for v in al_nz])
    check("c21.aligned_median_S2", float(med_s2), 1.10, 5e-3)

# ===========================================================================
# 2. 21 real cities, classic specification (contrast row of tab:threespec)
# ===========================================================================
print("\n=== 2. 21 cities, classic specification ===")
c21c = load(C21_CLASSIC)["results"]
poa21c = np.array([v["PoA"] for v in c21c.values()])
check("c21c.mean_PoA", float(poa21c.mean()), 1.0503, 5e-5)
check("c21c.median_PoA", float(np.median(poa21c)), 1.0386, 5e-5)
check("c21c.max_PoA", float(poa21c.max()), 1.1604, 5e-5)
check("c21c.violations_1.667", int((poa21c > 1.667).sum()), 0)

# ===========================================================================
# 3. Full enumeration, classic specification (Sec. 6, tab:enum_dist, tab:layered)
# ===========================================================================
print("\n=== 3. Full enumeration (n = 15,207) ===")
de = load(ENUM_EXACT)
recs = [r for r in de["results"] if r["nash_F2"] > 0]
milp = np.array([r["milp_F2"] for r in recs])
nash = np.array([r["nash_F2"] for r in recs])
m1 = np.array([r["milp_F1"] for r in recs])
n1 = np.array([r["nash_F1"] for r in recs])
s2m = np.array([r["milp_S2_weighted"] for r in recs])
s2n = np.array([r["nash_S2_weighted"] for r in recs])
cha = np.array([r["ch_assignment"] for r in recs])
chc = np.array([r["ch_congestion"] for r in recs])
chv = np.array([r["ch_coverage"] for r in recs])
PoA = milp / nash
dF2 = milp - nash
dcov = m1 - n1
dA2 = s2m - s2n
locked = dcov <= 1e-9
N = len(recs)

check("enum.n", N, 15207)
check("enum.mean_PoA", float(PoA.mean()), 1.0610, 5e-5)
check("enum.median_PoA", float(np.median(PoA)), 1.0372, 5e-5)
check("enum.p99_PoA", float(np.percentile(PoA, 99)), 1.3071, 5e-5)
check("enum.max_PoA", float(PoA.max()), 1.6404, 5e-5)
check("enum.min_PoA", float(PoA.min()), 0.9187, 5e-5)

eq = np.abs(PoA - 1.0) <= TOL_TIE
win = PoA > 1.0 + TOL_TIE
lose = PoA < 1.0 - TOL_TIE
check("enum.pct_MILP_strict", float(100 * win.mean()), 73.98, 0.01, "%.2f")
check("enum.pct_tie", float(100 * eq.mean()), 17.84, 0.01, "%.2f")
check("enum.n_tie", int(eq.sum()), 2713)
check("enum.pct_Nash_strict", float(100 * lose.mean()), 8.18, 0.01, "%.2f")
check("enum.n_Nash_strict", int(lose.sum()), 1244)
check("enum.smallest_nonzero_PoA_dev",
      float(np.abs(PoA[~eq] - 1.0).min()), 1.67e-5, 2e-7, "%.2e")
check("enum.violations_1.667", int((PoA > 1.667).sum()), 0)

check("enum.locked_n", int(locked.sum()), 4317)
check("enum.locked_share_pct", float(100 * locked.mean()), 28.4, 0.05, "%.1f")
check("enum.locked_mean", float(PoA[locked].mean()), 1.0092, 5e-5)
check("enum.locked_p99", float(np.percentile(PoA[locked], 99)), 1.0938, 5e-5)
check("enum.locked_max", float(PoA[locked].max()), 1.1560, 5e-5)
check("enum.unlocked_n", int((~locked).sum()), 10890)
check("enum.unlocked_mean", float(PoA[~locked].mean()), 1.0816, 5e-5)
check("enum.unlocked_p99", float(np.percentile(PoA[~locked], 99)), 1.3277, 5e-5)
check("enum.unlocked_max", float(PoA[~locked].max()), 1.6404, 5e-5)

non_tie = dF2 != 0
check("enum.non_tie_n", int(non_tie.sum()), 12495)
check("enum.channel_median_S1", float(100 * np.median(cha[non_tie] / dF2[non_tie])), 69.3, 0.05, "%.1f")
check("enum.channel_median_S2", float(100 * np.median(chc[non_tie] / dF2[non_tie])), -12.6, 0.05, "%.1f")
check("enum.channel_median_cov", float(100 * np.median(chv[non_tie] / dF2[non_tie])), 43.1, 0.05, "%.1f")
check("enum.identity_resid_max", float(np.max(np.abs(cha + chc + chv - dF2))),
      0.0, 1e-15, "%.1e")

deg = np.abs(dA2[locked]) <= 1e-9
check("enum.locked_degenerate_n", int(deg.sum()), 3243)
check("enum.locked_degenerate_pct", float(100 * deg.mean()), 75.1, 0.05, "%.1f")
sub = locked & (np.abs(dA2) > 1e-9) & non_tie
check("enum.locked_nondegenerate_nontie_n", int(sub.sum()), 1074)
check("enum.locked_nondegen_median_S1", float(100 * np.median(cha[sub] / dF2[sub])), 28.6, 0.05, "%.1f")
check("enum.locked_nondegen_median_S2", float(100 * np.median(chc[sub] / dF2[sub])), 71.4, 0.05, "%.1f")

X = np.column_stack([np.ones(N), locked.astype(float)])
beta = np.linalg.lstsq(X, PoA, rcond=None)[0]
r2_lock = 1 - ((PoA - X @ beta) ** 2).sum() / ((PoA - PoA.mean()) ** 2).sum()
check("enum.locked_dummy_R2", float(r2_lock), 0.194, 5e-4)

# share of configurations above the conditional bound of prop:conditional (1.20)
check("enum.n_above_bound_1.20", int((PoA > 1.20).sum()), 858)
check("enum.pct_above_bound_1.20", float(100 * (PoA > 1.20).mean()), 5.64, 0.01, "%.2f")

# ===========================================================================
# 4. Satisfaction-floor sweep (tab:sensitivity)
# ===========================================================================
print("\n=== 4. theta sweep ===")
THETA_EXP = {"0.50": (1.2174, 1.8798, 1.0502),
             "0.60": (1.1560, 1.6404, 1.0610),
             "0.70": (1.1200, 1.6017, 1.0680)}
for t, (lmax, gmax, mean) in THETA_EXP.items():
    p = THETA_FILES[t]
    if not os.path.exists(p):
        print("  [SKIP] theta=%s: %s absent" % (t, os.path.relpath(p, ROOT)))
        continue
    rr = load(p)["results"]
    pm = np.array([r["milp_F2"] for r in rr])
    pn = np.array([r["nash_F2"] for r in rr])
    p1m = np.array([r["milp_F1"] for r in rr])
    p1n = np.array([r["nash_F1"] for r in rr])
    p_poa = pm / pn
    lk = (p1m - p1n) <= 1e-9
    check("theta%s.locked_max" % t, float(p_poa[lk].max()), lmax, 5e-5)
    check("theta%s.global_max" % t, float(p_poa.max()), gmax, 5e-5)
    check("theta%s.mean" % t, float(p_poa.mean()), mean, 5e-5)
    check("theta%s.locked_violations" % t,
          int((p_poa[lk] > 1.0 / float(t) + 1e-9).sum()), 0)

# theta = 0.60 must reproduce the primary enumeration configuration by configuration
if os.path.exists(THETA_FILES["0.60"]):
    by_cfg = {tuple(r["config"]): r for r in load(THETA_FILES["0.60"])["results"]}
    worst = max(abs(r["milp_F2"] - by_cfg[tuple(r["config"])]["milp_F2"])
                + abs(r["nash_F2"] - by_cfg[tuple(r["config"])]["nash_F2"]) for r in recs)
    check("theta0.60.config_by_config_max_diff", float(worst), 0.0, 0.0, "%.1e")

# ===========================================================================
# 5. Subsidy (Sec. 7)
# ===========================================================================
print("\n=== 5. Subsidy ===")
ds = load(SUBSIDY)
sr = list(ds["results"].values()) if isinstance(ds["results"], dict) else ds["results"]
check("subsidy.n", len(sr), 21)
check("subsidy.mean_improve_pct",
      float(np.mean([r["nash_F2_improve_pct"] for r in sr])), 5.97, 5e-3, "%.2f")
check("subsidy.mean_PoA_base", float(np.mean([r["PoA_base"] for r in sr])), 1.0503, 5e-5)
check("subsidy.mean_PoA_subsidized",
      float(np.mean([r["PoA_sub_consistent"] for r in sr])), 1.0656, 5e-5)

# ===========================================================================
# 6. City-level regression and leave-one-out (tab:reg)
# ===========================================================================
print("\n=== 6. Regression (tab:reg) ===")


def capacity_gini(nl, nm, ns):
    caps = np.sort(np.array([3000.0] * int(nl) + [2000.0] * int(nm) + [1000.0] * int(ns)))
    n = len(caps)
    return 2.0 * np.sum(np.arange(1, n + 1) * caps) / (n * caps.sum()) - (n + 1) / n


def frame(res):
    rows = [dict(city=k, n_stations=int(v["n_stations"]),
                 capacity_gini=round(capacity_gini(v["n_large"], v.get("n_medium", 0), v["n_small"]), 4),
                 connectivity=float(v["connectivity"]), cluster_r=float(v["cluster_r"]),
                 PoA=float(v["PoA"])) for k, v in res.items()]
    df = pd.DataFrame(rows)
    df["log_n_stations"] = np.log(df["n_stations"])
    return df


XCOLS = ["connectivity", "capacity_gini", "log_n_stations", "cluster_r"]
df = frame(c21)
Xr = sm.add_constant(df[XCOLS])
mfit = sm.OLS(df["PoA"], Xr).fit()
check("reg.R2", float(mfit.rsquared), 0.70247, 5e-5, "%.5f")
check("reg.AdjR2", float(mfit.rsquared_adj), 0.62809, 5e-5, "%.5f")
check("reg.F_p", float(mfit.f_pvalue), 0.000407, 1e-6, "%.6f")
check("reg.beta_connectivity", float(mfit.params["connectivity"]), 2.1523, 5e-4)
check("reg.p_connectivity", float(mfit.pvalues["connectivity"]), 0.1229, 5e-4)
check("reg.ci_low_connectivity", float(mfit.conf_int().loc["connectivity", 0]), -0.6491, 5e-4)
check("reg.ci_high_connectivity", float(mfit.conf_int().loc["connectivity", 1]), 4.9537, 5e-4)
check("reg.beta_capacity_gini", float(mfit.params["capacity_gini"]), 0.1278, 5e-4)
check("reg.beta_log_n_stations", float(mfit.params["log_n_stations"]), 0.024579, 5e-6, "%.6f")
check("reg.p_log_n_stations", float(mfit.pvalues["log_n_stations"]), 0.0023, 5e-5)

Zf = df[XCOLS + ["PoA"]]
Z = (Zf - Zf.mean()) / Zf.std()
mstd = sm.OLS(Z["PoA"], sm.add_constant(Z[XCOLS])).fit()
check("reg.std_beta_connectivity", float(mstd.params["connectivity"]), 0.4773, 5e-4)
check("reg.std_beta_capacity_gini", float(mstd.params["capacity_gini"]), 0.0289, 5e-4)
check("reg.std_beta_log_n_stations", float(mstd.params["log_n_stations"]), 0.6373, 5e-4)
check("reg.std_beta_cluster_r", float(mstd.params["cluster_r"]), 0.0581, 5e-4)
vif = {c: variance_inflation_factor(Xr.values, i) for i, c in enumerate(Xr.columns)}
check("reg.vif_connectivity", float(vif["connectivity"]), 4.619, 5e-3)
check("reg.vif_max_predictors",
      float(max(v for k, v in vif.items() if k != "const")), 4.685, 5e-3)

loo_coef, loo_sig, loo_fsig = [], 0, 0
for city in df["city"]:
    dsub = df[df["city"] != city]
    msub = sm.OLS(dsub["PoA"], sm.add_constant(dsub[XCOLS])).fit()
    loo_coef.append(msub.params["log_n_stations"])
    loo_sig += int(msub.pvalues["log_n_stations"] < 0.05)
    loo_fsig += int(msub.f_pvalue < 0.05)
check("loo.n_significant_log_n", loo_sig, 21)
check("loo.n_significant_model", loo_fsig, 21)
check("loo.coef_min", float(min(loo_coef)), 0.0175, 5e-4)
check("loo.coef_max", float(max(loo_coef)), 0.0363, 5e-4)

# --------------------------------------------------------------- cluster radius
# The three cluster-radius factors are re-run with the exact Phase 2; the paper
# quotes the mean of each cell, the worst single-city PoA over all three cells,
# and the number of cities that reach solver optimality.
cf_means = {}
cf_nonopt = 0
cf_poas = []
for cf in ("0.8", "1.0", "1.2"):
    path = os.path.join(ROOT, "results", "main",
                        "21cities_exact_locked.json" if cf == "1.0"
                        else "21cities_cf%s_exact_locked.json" % cf)
    if not os.path.exists(path):
        continue
    cell = json.load(open(path, encoding="utf-8"))["results"]
    cf_means[cf] = float(np.mean([v["PoA"] for v in cell.values()]))
    cf_nonopt += sum(1 for v in cell.values() if v.get("milp_status") != "Optimal")
    cf_poas.extend(v["PoA"] for v in cell.values())
check("cf.mean_0.8", cf_means.get("0.8", float("nan")), 1.0688)
check("cf.mean_1.0", cf_means.get("1.0", float("nan")), 1.0376)
check("cf.mean_1.2", cf_means.get("1.2", float("nan")), 1.0267)
check("cf.band_width", max(cf_means.values()) - min(cf_means.values()), 0.0421, 1e-3)
check("cf.max_over_cells", float(max(cf_poas)) if cf_poas else float("nan"), 1.1764)
check("cf.n_nonoptimal", cf_nonopt, 0)

# ------------------------------------------------- Shenzhen exclusion
# "Removing Shenzhen moves the mean only from 1.0376 to 1.0395."
check("shenzhen.mean_excl",
      float(np.mean([v["PoA"] for k, v in c21.items() if k != "Shenzhen"])), 1.0395)

dfc = frame(c21c)
mcl = sm.OLS(dfc["PoA"], sm.add_constant(dfc[XCOLS])).fit()
check("reg_classic.R2", float(mcl.rsquared), 0.0777, 5e-4)
check("reg_classic.AdjR2", float(mcl.rsquared_adj), -0.1529, 5e-4)
check("reg_classic.F_p", float(mcl.f_pvalue), 0.849, 5e-3)

# ===========================================================================
# Summary
# ===========================================================================
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
json.dump(OUT, open(OUT_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("\nrecomputed values ->", os.path.relpath(OUT_PATH, ROOT))
if FAILS:
    print("\n=== FAILED (%d) ===" % len(FAILS))
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("\nALL CHECKS PASSED (%d numbers match the manuscript)" % len(OUT))
