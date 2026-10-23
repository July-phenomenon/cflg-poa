# -*- coding: utf-8 -*-
"""Enumeration connectivity regression (full 15,207-config space).

Tests the competing hypothesis that the PoA is driven by network topology rather
than by the satisfaction architecture: regresses the measured PoA on the
connectivity index

    connectivity = mean over communities of (#stations within 1000 m) / n_sta

evaluated on the same synthetic geometry that generated the instances.

Input : results/main/results_enumeration_exact.json   (or the fixed-point file)
Output: results/tables/T11_enum_connectivity_reg.json

The geometry is rebuilt from the shipped generator (seed 2024, side 2200 m) and
each configuration's station coordinates are the community coordinates whose tier
is non-zero -- exactly what ``make_config`` does when solving.
"""
import json
import os
import sys

import numpy as np
import statsmodels.api as sm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "02_core_solver")))

from instance_generator import (base_geometry, make_config,  # noqa: E402
                                enumerate_configs, distance_matrix)

PKG = os.path.abspath(os.path.join(_HERE, "..", ".."))
SEED = 2024
SIDE = 2200.0
RADIUS = 1000.0


def main():
    src = os.environ.get(
        "CFLG_ENUM_FILE",
        os.path.join(PKG, "results", "main", "results_enumeration_exact.json"))
    with open(src, encoding="utf-8") as f:
        recs = json.load(f)["results"]
    by_cfg = {tuple(int(t) for t in r["config"]): r for r in recs}
    print("loaded %d records from %s" % (len(by_cfg), os.path.basename(src)))

    xy, pop, hetero = base_geometry(seed=SEED, side=SIDE)
    configs = enumerate_configs(budget=None)
    print("enumerated %d configurations" % len(configs))
    assert len(configs) == 15207, "config space drifted: %d" % len(configs)

    conn, poa, nsta = [], [], []
    for tiers in configs:
        inst = make_config(tiers, xy=xy, pop=pop, hetero=hetero)
        n_sta = len(inst["built"])
        if n_sta == 0:
            continue
        d = distance_matrix(xy, xy[inst["built"]])
        conn.append(float(np.mean((d <= RADIUS).sum(axis=1)) / n_sta))
        poa.append(float(by_cfg[tuple(int(t) for t in tiers)]["PoA"]))
        nsta.append(n_sta)

    conn = np.asarray(conn)
    poa = np.asarray(poa)
    print("n=%d  connectivity in [%.4f, %.4f]" % (len(poa), conn.min(), conn.max()))

    X = sm.add_constant(conn)
    fit = sm.OLS(poa, X).fit()
    beta = float(fit.params[1])
    pval = float(fit.pvalues[1])
    r2 = float(fit.rsquared)
    rho = float(np.corrcoef(conn, poa)[0, 1])
    print("beta_connectivity = %.6f   p = %.4f   R^2 = %.6f   corr = %.4f"
          % (beta, pval, r2, rho))

    # robustness: control for network size, which co-varies with connectivity
    Xc = sm.add_constant(np.column_stack([conn, np.log(np.asarray(nsta))]))
    fit_c = sm.OLS(poa, Xc).fit()
    print("controlled: beta_conn = %.6f (p = %.4f), beta_log_nsta = %.6f (p = %.4f)"
          % (fit_c.params[1], fit_c.pvalues[1], fit_c.params[2], fit_c.pvalues[2]))

    out = dict(
        source=os.path.basename(src),
        n=int(len(poa)),
        geometry=dict(seed=SEED, side_m=SIDE, radius_m=RADIUS),
        simple=dict(beta_connectivity=round(beta, 6),
                    p_value=round(pval, 4),
                    r_squared=round(r2, 6),
                    correlation=round(rho, 4)),
        controlled=dict(beta_connectivity=round(float(fit_c.params[1]), 6),
                        p_connectivity=round(float(fit_c.pvalues[1]), 4),
                        beta_log_n_sta=round(float(fit_c.params[2]), 6),
                        p_log_n_sta=round(float(fit_c.pvalues[2]), 4),
                        r_squared=round(float(fit_c.rsquared), 6)),
    )
    dst = os.path.join(PKG, "results", "tables", "T11_enum_connectivity_reg.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("saved ->", dst)


if __name__ == "__main__":
    main()
