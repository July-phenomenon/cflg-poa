# -*- coding: utf-8 -*-
"""Enumeration regression ladder: connectivity vs coverage-lock (Appendix C audit).

Why this script exists
----------------------
Section 6.7 of the manuscript compares two explanations of the measured PoA over
the full 15,207-configuration space:

    "connectivity ... beta = 0.152, R^2 = 0.0069 ...
     the coverage-lock indicator alone explains 19.4%,
     adding connectivity raises R^2 from 0.194 to 0.200;
     adding log n_sta brings it to 0.204."

``exp2_enum_connectivity_reg.py`` releases the *first* model only (T11).  The
nested R^2 chain (0.194 -> 0.200 -> 0.204) and the lock-only model were never
released as an artifact, so they could not be audited.  This script re-estimates
the whole ladder on the shipped enumeration output and writes every coefficient,
so each quoted number has a machine-readable source.

Models (dependent variable: measured PoA)
-----------------------------------------
    M0   PoA ~ 1                      (baseline)
    M1   PoA ~ connectivity           (the "geographic" hypothesis)
    M2   PoA ~ locked                 (the "structural" hypothesis)
    M3   PoA ~ locked + connectivity
    M4   PoA ~ locked + connectivity + log n_sta

``locked`` is the coverage-lock indicator: |milp_F1 - nash_F1| <= 1e-9 on the
same configuration.  ``connectivity`` is recomputed on the shipped synthetic
geometry (seed 2024, side 2200 m, radius 1000 m), exactly as in
``exp2_enum_connectivity_reg.py``.

Inputs
------
``results/main/results_enumeration_exact.json``  (override: $CFLG_ENUM_FILE)

Outputs
-------
``results/tables/T15_enum_regression_ladder.json``
``results/tables/T15_enum_regression_ladder.md``   (human-readable audit trail)

Usage
-----
    python code/03_experiments/exp2_enum_regression_full.py
"""
import json
import os
import sys

import numpy as np
import statsmodels.api as sm

_HERE = os.path.dirname(os.path.abspath(__file__))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "02_core_solver")))

from instance_generator import (base_geometry, make_config,  # noqa: E402
                                enumerate_configs, distance_matrix)

PKG = os.path.abspath(os.path.join(_HERE, "..", ".."))
SEED, SIDE, RADIUS = 2024, 2200.0, 1000.0


def build_covariates(src):
    with open(src, encoding="utf-8") as f:
        recs = json.load(f)["results"]
    by_cfg = {tuple(int(t) for t in r["config"]): r for r in recs}
    print("loaded %d records from %s" % (len(by_cfg), os.path.basename(src)))

    xy, pop, hetero = base_geometry(seed=SEED, side=SIDE)
    configs = enumerate_configs(budget=None)
    assert len(configs) == 15207, "config space drifted: %d" % len(configs)

    conn, poa, nsta, lock = [], [], [], []
    for i, tiers in enumerate(configs):
        inst = make_config(tiers, xy=xy, pop=pop, hetero=hetero)
        n_sta = len(inst["built"])
        if n_sta == 0:
            continue
        d = distance_matrix(xy, xy[inst["built"]])
        rec = by_cfg[tuple(int(t) for t in tiers)]
        conn.append(float(np.mean((d <= RADIUS).sum(axis=1)) / n_sta))
        poa.append(float(rec["PoA"]))
        nsta.append(n_sta)
        lock.append(1.0 if abs(rec["milp_F1"] - rec["nash_F1"]) <= 1e-9 else 0.0)
    return (np.asarray(conn), np.asarray(poa), np.asarray(nsta, dtype=float),
            np.asarray(lock))


def fit(y, cols):
    X = sm.add_constant(np.column_stack(cols), has_constant="add")
    m = sm.OLS(y, X).fit()
    return dict(params=[float(v) for v in m.params],
                pvalues=[float(v) for v in m.pvalues],
                r_squared=float(m.rsquared),
                adj_r_squared=float(m.rsquared_adj),
                n=int(m.nobs))


def main():
    src = os.environ.get(
        "CFLG_ENUM_FILE",
        os.path.join(PKG, "results", "main", "results_enumeration_exact.json"))
    conn, poa, nsta, lock = build_covariates(src)
    logn = np.log(nsta)
    print("n=%d  locked=%.1f%%  conn in [%.3f, %.3f]"
          % (len(poa), 100 * lock.mean(), conn.min(), conn.max()))

    models = {
        "M1_connectivity": fit(poa, [conn]),
        "M2_locked": fit(poa, [lock]),
        "M3_locked_connectivity": fit(poa, [lock, conn]),
        "M4_locked_connectivity_lognsta": fit(poa, [lock, conn, logn]),
    }
    for name, m in models.items():
        print("%-32s R2=%.4f  adjR2=%.4f" % (name, m["r_squared"],
                                             m["adj_r_squared"]))

    out = dict(source=os.path.basename(src), n=len(poa),
               geometry=dict(seed=SEED, side_m=SIDE, radius_m=RADIUS),
               locked_share=float(lock.mean()),
               models=models,
               notes=("connectivity = mean over communities of "
                      "(#stations within 1000 m)/n_sta, rebuilt from the "
                      "shipped generator; locked = |milp_F1-nash_F1|<=1e-9"))
    dst = os.path.join(PKG, "results", "tables", "T15_enum_regression_ladder.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    lines = ["# Enumeration regression ladder (audit trail)", "",
             "source: `%s`, n = %d, locked share = %.1f%%"
             % (out["source"], out["n"], 100 * out["locked_share"]), "",
             "| model | R2 | adj. R2 | coefficients (p) |", "|---|---|---|---|"]
    names = {"M1_connectivity": ["const", "connectivity"],
             "M2_locked": ["const", "locked"],
             "M3_locked_connectivity": ["const", "locked", "connectivity"],
             "M4_locked_connectivity_lognsta":
                 ["const", "locked", "connectivity", "log n_sta"]}
    for k, m in models.items():
        cells = ", ".join("%s = %.4g (%.3g)" % (n_, b, p)
                          for n_, b, p in zip(names[k], m["params"],
                                              m["pvalues"]))
        lines.append("| %s | %.4f | %.4f | %s |"
                     % (k, m["r_squared"], m["adj_r_squared"], cells))
    with open(os.path.join(PKG, "results", "tables",
                           "T15_enum_regression_ladder.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("saved ->", dst)


if __name__ == "__main__":
    main()
