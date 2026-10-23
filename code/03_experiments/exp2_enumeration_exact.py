# -*- coding: utf-8 -*-
"""§6 full enumeration with an EXACT Phase-2 MILP (P0-3, option B).

Background
----------
The published enumeration (`results_enumeration.json`) computes the centralized
Phase-2 objective with `core_solver.run_milp_exact`, which freezes the
utilisation satisfactions S2, solves for the assignment, recomputes the induced
utilisation, and repeats (<= 10 fixed-point rounds).  That value carries no
global-optimality guarantee, so every PoA in the paper is only a lower bound, and
the heuristic can occasionally be beaten (measured: up to +2.3e-3 in F2).

This script recomputes Phase 2 with the exact single-MILP encoding in
`core_solver_exact` (staircase segments as binaries + McCormick linearisation of
z * population-served).  The optimality certificate now holds, so PoA becomes a
genuine ratio rather than a lower bound.  A brute-force gold-standard check
(`_bruteforce_check.py`) agrees with the exact solver to 0.00e+00.

Nash values are NOT recomputed: the decentralized equilibrium is a different
model (best-response dynamics) and is unaffected by the Phase-2 fix.  Its F2/F1/S2
are copied from the published file so that both runs are compared on identical
denominators.

Solver
------
HiGHS by default.  CBC (via PuLP) is known to return *false* Infeasible on this
model family and to occasionally report a suboptimal point as Optimal, so it is
used only as a fallback, never as the primary.

Output
------
results/main/results_enumeration_exact.json
    {geometry, solver, source, results:[...]}
Each record carries the exact figures plus the previous heuristic ones for audit:
    config, nash_F2, nash_F1, nash_S2_weighted,
    milp_F2, milp_F1, milp_S2_weighted, PoA,
    ch_assignment, ch_congestion, ch_coverage,
    milp_F2_heur, PoA_heur, status
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_UTILS = os.path.abspath(os.path.join(_HERE, "..", "utils"))
_SOLVER = os.path.abspath(os.path.join(_HERE, "..", "02_core_solver"))
for _p in (_HERE, _UTILS, _SOLVER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from constants import W1, W2, W3, S3, PKG_ROOT                     # noqa: E402
from instance_generator import (base_geometry, enumerate_configs,   # noqa: E402
                                make_config)
from core_solver_exact import run_two_phase_exact                   # noqa: E402


def results_dir(*parts):
    return os.path.join(PKG_ROOT, "results", *parts)


def log(msg, fh=None):
    line = "[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def solve_one(tiers, geo, prev):
    """Exact Phase-2 for one configuration; Nash side copied from ``prev``."""
    inst = make_config(tiers, xy=geo[0], pop=geo[1], hetero=geo[2])
    n, pop, demand, S1m, caps = (inst["n"], inst["pop"], inst["demand"],
                                 inst["S1m"], inst["sta_caps"])

    nash_F2 = float(prev["nash_F2"])
    nash_F1 = float(prev["nash_F1"])
    nash_S2 = float(prev["nash_S2_weighted"])
    if nash_F2 <= 0:
        return None

    milp_F2, milp_F1, _cov, _w, milp_S2, info = run_two_phase_exact(
        n, pop, demand, S1m, caps, return_details=True,
        solver="highs", fallback="cbc")
    if not info:
        return None
    s2_mean = float(np.mean(milp_S2)) if len(milp_S2) else 0.0

    # --- channel decomposition of ΔF2 (§6.3), identical algebra to the
    #     published run so the two are directly comparable --------------------
    d_cov = milp_F1 - nash_F1
    d_A2 = s2_mean - nash_S2
    A1_n = (nash_F2 - W2 * nash_S2 - W3 * S3 * nash_F1) / W1
    A1_m = (milp_F2 - W2 * s2_mean - W3 * S3 * milp_F1) / W1
    d_A1 = A1_m - A1_n

    return {
        "config": list(map(int, tiers)),
        "nash_F2": nash_F2,
        "milp_F2": float(milp_F2),
        "PoA": float(milp_F2 / nash_F2),
        "nash_F1": nash_F1,
        "milp_F1": float(milp_F1),
        "nash_S2_weighted": nash_S2,
        "milp_S2_weighted": s2_mean,
        "ch_assignment": float(W1 * d_A1),
        "ch_congestion": float(W2 * d_A2),
        "ch_coverage": float(W3 * S3 * d_cov),
        "milp_F2_heur": float(prev["milp_F2"]),
        "PoA_heur": float(prev["PoA"]),
        "status": info.get("status"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="§6 enumeration with exact Phase-2")
    ap.add_argument("--limit", type=int, default=0, help="first N configs only")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--side", type=float, default=2200.0)
    ap.add_argument("--source", default=None,
                    help="published (heuristic) enumeration to compare against")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    os.makedirs(results_dir("main"), exist_ok=True)
    os.makedirs(results_dir("logs"), exist_ok=True)
    src_path = args.source or results_dir("main", "results_enumeration.json")
    out_path = args.out or results_dir("main", "results_enumeration_exact.json")
    log_path = results_dir("logs", "exp2_exact_run.log")
    geom_meta = {"seed": args.seed, "side_m": args.side}

    src = json.load(open(src_path, encoding="utf-8"))
    if src.get("geometry") != geom_meta:
        raise SystemExit("source geometry %r != requested %r"
                         % (src.get("geometry"), geom_meta))
    prev_by_key = {tuple(r["config"]): r for r in src["results"]}
    log("loaded %d published records from %s" % (len(prev_by_key), src_path))

    geo = base_geometry(seed=args.seed, side=args.side)
    configs = enumerate_configs(budget=None)
    assert len(configs) == 15207, "config space drifted (%d)" % len(configs)
    if args.limit:
        configs = configs[:args.limit]

    fh = open(log_path, "w", encoding="utf-8")
    log("exact Phase-2 enumeration: %d configurations  geo(seed=%d, side=%.0f)"
        % (len(configs), args.seed, args.side), fh)
    log("solver=HiGHS (fallback CBC)   output -> %s" % out_path, fh)

    records, t0 = [], time.time()
    done_keys = set()
    if os.path.exists(out_path):
        try:
            _prev_out = json.load(open(out_path, encoding="utf-8"))
            if _prev_out.get("geometry") == geom_meta:
                done_keys = {tuple(r["config"]) for r in _prev_out.get("results", [])}
                records = list(_prev_out["results"])
                log("resumed: %d configs already done" % len(done_keys), fh)
        except Exception as exc:
            log("cannot resume from %s (%r) -> fresh run" % (out_path, exc), fh)

    def _flush():
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"geometry": geom_meta, "solver": "highs",
                       "phase1_solver": "highs",
                       "source": os.path.basename(src_path),
                       "results": records}, f)
        os.replace(tmp, out_path)

    n_skip = n_fail = 0
    for i, tiers in enumerate(configs, 1):
        key = tuple(tiers)
        if key in done_keys:
            n_skip += 1
            continue
        prev = prev_by_key.get(key)
        if prev is None:
            n_fail += 1
            log("config %d has no published counterpart -> skipped" % i, fh)
            continue
        try:
            rec = solve_one(tiers, geo, prev)
        except Exception as exc:
            log("config %d failed: %r" % (i, exc), fh)
            rec = None
        if rec is None:
            n_fail += 1
        else:
            records.append(rec)
        if i % 500 == 0 or i == len(configs):
            _poa = [r["PoA"] for r in records]
            log("%d/%d  done=%d  fail=%d  mean PoA=%.4f  elapsed=%.0fs"
                % (i, len(configs), len(records), n_fail,
                   float(np.mean(_poa)) if _poa else 0.0, time.time() - t0), fh)
            _flush()

    _flush()
    poa = np.array([r["PoA"] for r in records])
    d = np.array([r["milp_F2"] - r["milp_F2_heur"] for r in records])
    log("done: %d records in %.1f s (skipped %d, failed %d)"
        % (len(records), time.time() - t0, n_skip, n_fail), fh)
    if len(poa):
        log("mean PoA %.4f  median %.4f  p99 %.4f  max %.4f"
            % (poa.mean(), np.median(poa), np.percentile(poa, 99), poa.max()), fh)
        log("exact-minus-heuristic F2: min %+.3e  mean %+.3e  max %+.3e  (n_improved %d)"
            % (d.min(), d.mean(), d.max(), int((d > 1e-9).sum())), fh)
        log("configs where heuristic beat the exact optimum (should be 0): %d"
            % int((d < -1e-9).sum()), fh)
    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
