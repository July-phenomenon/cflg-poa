# -*- coding: utf-8 -*-
"""Satisfaction-ladder refinement test (paper P2-1 / review point A12).

The question this script answers
--------------------------------
Reviewers (independently, in two rounds of simulated review) raised the same
"simpler explanation" for the small measured PoA:

    the utilisation-satisfaction ladder S2 has only FIVE levels
    (1.00 / 0.93 / 0.85 / 0.72 / 0.60).  With such a coarse staircase the
    Nash objective and the MILP objective can only differ in coarse
    increments, so a large share of configurations tie (17.8% have
    PoA exactly 1) and the ratio is structurally pinned near one.

If that is the whole story, refining the ladder should make the ties
disappear AND the measured PoA rise markedly.  If the PoA stays put while
the ties collapse, the small ratio is a property of the game rather than of
the discretisation.

What is varied
--------------
ONLY the granularity of the S2 staircase.  Specifically:

  * the payoff RANGE is held fixed at [0.60, 1.00] -- the two endpoints of
    the published ladder.  Theorem 1's bound is 1/S_min = 1/0.60 = 1.667 and
    depends only on that range, so the theoretical bound is IDENTICAL across
    every ladder; only the MEASURED ratio can move.  This is the cleanest
    possible falsification test: it isolates granularity from range.
  * S1 (distance satisfaction, 4 levels) and S3 (constant 0.60) are held
    fixed.  Neither is a strategic variable: S1 depends only on the fixed
    geometry, and S3 is a constant.  S2 -- the congestion-feedback term -- is
    the only component that carries strategic content, so it is the right
    target and the only one worth refining.

Construction of the refined ladders
-----------------------------------
The published ladder is read as a staircase sampling of an underlying
decreasing curve g at its breakpoints:

    g(0.60)=1.00, g(0.75)=0.93, g(0.85)=0.85, g(0.95)=0.72, g(1.00)=0.60
    (segment (u_{k-1}, u_k] carries the value g(u_k); u below 0.60 -> 1.00)

A K-level ladder keeps the first breakpoint at 0.60 (preserving the flat
region where an under-used station earns full satisfaction) and places the
remaining K-1 breakpoints evenly over (0.60, 1.00], with values read off the
linear interpolant of the five anchor points above.  Every refined ladder is
therefore the SAME function approximated more finely -- not a different
function.  Monotonicity and the [0.60, 1.00] range are hard-asserted.

K = 5 is a special case: it is the PUBLISHED ladder, used verbatim as the
control arm, and is therefore not the even-spacing 5-level approximation.

Control design (why we re-run K=5 instead of reusing the archive)
-----------------------------------------------------------------
The archived full enumeration (`results/main/results_enumeration_exact.json`)
was produced with the same geometry seed (2024) and side (2200 m), but the
theta sweep of 2026-10-21 showed that in-process regeneration does not
reproduce the archive bit-for-bit (geometry-calibration caveat).  A refined
arm must therefore be compared against a K=5 arm run **in the same process,
on the same geometry, through the same code path** -- only then is the ladder
the single changed variable.  `--control N` quantifies the residual
process-vs-archive drift so that it can be reported honestly; the scientific
comparison is always arm-vs-arm within one run.

Usage
-----
    python ladder_refinement.py --control 60          # control vs archive
    python ladder_refinement.py --ladder 5            # published ladder (arm 0)
    python ladder_refinement.py --ladder 20           # refined arm
    python ladder_refinement.py --ladder 100          # near-continuum limit
    python ladder_refinement.py --all                 # 5,10,20,50,100 sequential

Output
------
    results/sensitivity/ladder_K<K>_per_config.json
    results/logs/ladder_K<K>.log
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from datetime import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOLVER = os.path.abspath(os.path.join(_HERE, "..", "02_core_solver"))
_UTILS = os.path.abspath(os.path.join(_HERE, "..", "utils"))
for _p in (_SOLVER, _UTILS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import core_solver                      # noqa: E402
import constants                         # noqa: E402
from constants import W1, W2, W3        # noqa: E402
from instance_generator import (base_geometry, enumerate_configs,   # noqa: E402
                                make_config)
from core_solver import run_nash        # noqa: E402

# --------------------------------------------------------------------------
# The published ladder, and the curve it samples
# --------------------------------------------------------------------------
PUBLISHED = [(0.6, 1.0), (0.75, 0.93), (0.85, 0.85), (0.95, 0.72), (1.0, 0.6)]
_U_ANCHOR = [p[0] for p in PUBLISHED]
_V_ANCHOR = [p[1] for p in PUBLISHED]
FLOOR = 0.60
CEIL = 1.00

ARCHIVE = os.path.join(constants.PKG_ROOT, "results", "main",
                       "results_enumeration_exact.json")


def refine_ladder(K):
    """K-level staircase sampling the same underlying curve as PUBLISHED."""
    if int(K) == 5:
        return [tuple(p) for p in PUBLISHED]
    K = int(K)
    if K < 3:
        raise ValueError("K must be >= 3 (got %r)" % K)
    us = [FLOOR] + [FLOOR + (k - 1) * (CEIL - FLOOR) / (K - 1)
                    for k in range(2, K + 1)]
    vs = [float(np.interp(u, _U_ANCHOR, _V_ANCHOR)) for u in us]
    lad = [(float(u), float(v)) for u, v in zip(us, vs)]
    _check_ladder(lad, K)
    return lad


def _check_ladder(lad, K):
    """Hard-assert the invariants a valid refinement must satisfy."""
    assert len(lad) == K, "ladder length %d != K %d" % (len(lad), K)
    us = [u for u, _ in lad]
    vs = [v for _, v in lad]
    assert us == sorted(us), "breakpoints not increasing"
    assert len(set(us)) == K, "duplicate breakpoints"
    assert all(vs[i] >= vs[i + 1] - 1e-15 for i in range(K - 1)), \
        "ladder not monotone decreasing"
    assert abs(vs[0] - CEIL) < 1e-9, "ceiling %r != 1.00" % vs[0]
    assert abs(vs[-1] - FLOOR) < 1e-9, "floor %r != 0.60" % vs[-1]
    assert abs(us[-1] - 1.0) < 1e-12, "last breakpoint must be 1.0"
    assert min(vs) >= FLOOR - 1e-12 and max(vs) <= CEIL + 1e-12, \
        "payoff range escaped [0.60, 1.00]"


def apply_ladder(lad):
    """Patch the shared staircase in place, then verify it was observed."""
    core_solver.S2_T[:] = [tuple(p) for p in lad]
    # --- the solvers MUST observe the patch ---------------------------------
    assert abs(core_solver.s2_fn(1.0) - lad[-1][1]) < 1e-12, "S2 floor not applied"
    assert abs(core_solver.s2_fn(0.60) - lad[0][1]) < 1e-12, "S2 ceiling not applied"
    # S1 must be untouched by a ladder experiment
    assert abs(core_solver.s1_fn(300.0) - 1.0) < 1e-12, "S1 inner tier corrupted"
    assert abs(core_solver.s1_fn(1000.0) - 0.60) < 1e-12, "S1 floor corrupted"
    assert abs(core_solver.BASELINE_S3 - 0.60) < 1e-12, "S3 baseline drifted"
    # Theorem 1 depends only on the range, which is invariant here
    s_min = W1 * 0.60 + W2 * 0.60 + W3 * 0.60
    assert abs(s_min - 0.60) < 1e-12, "S_min != theta"
    try:
        from core_solver_exact import sync_stairs
        sync_stairs()
    except ImportError:
        pass
    return dict(K=len(lad), ladder=[list(p) for p in lad],
                S1_T=list(map(list, core_solver.S1_T)),
                BASELINE_S3=float(core_solver.BASELINE_S3),
                S_min=s_min, theorem1_bound=1.0 / s_min)


def log(msg, fh=None):
    line = "[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def solve_one(tiers, geo, tag):
    """Solve one configuration through the exact two-phase path."""
    from core_solver_exact import run_two_phase_exact
    inst = make_config(tiers, xy=geo[0], pop=geo[1], hetero=geo[2])
    n, pop, demand, S1m, caps = (inst["n"], inst["pop"], inst["demand"],
                                 inst["S1m"], inst["sta_caps"])
    nash_F2, nash_F1, _ = run_nash(n, pop, demand, S1m, caps)
    milp_F2, milp_F1, _ = run_two_phase_exact(n, pop, demand, S1m, caps)
    if nash_F2 <= 0:
        return None
    return {
        "config": list(map(int, tiers)),
        "K": tag,
        "nash_F2": float(nash_F2),
        "milp_F2": float(milp_F2),
        "PoA": float(milp_F2 / nash_F2),
        "nash_F1": float(nash_F1),
        "milp_F1": float(milp_F1),
    }


def _worker(payload):
    tiers, geo, tag, lad = payload
    # Windows spawn re-imports the module with the DEFAULT ladder; re-apply in
    # every child and assert, so that no configuration is ever solved on the
    # wrong staircase.
    core_solver.S2_T[:] = [tuple(p) for p in lad]
    try:
        from core_solver_exact import sync_stairs
        sync_stairs()
    except ImportError:
        pass
    assert abs(core_solver.s2_fn(1.0) - lad[-1][1]) < 1e-12, "worker not patched"
    try:
        return solve_one(tiers, geo, tag)
    except Exception:
        return None


def summarise(records, label=""):
    p = [r["PoA"] for r in records if r and r.get("PoA") is not None]
    if not p:
        return {}
    s = {
        "n": len(p),
        "mean": round(st.mean(p), 4),
        "median": round(st.median(p), 4),
        "p95": round(sorted(p)[int(0.95 * len(p)) - 1], 4),
        "min": round(min(p), 4),
        "max": round(max(p), 4),
        "frac_tie": round(sum(1 for x in p if abs(x - 1.0) < 1e-12) / len(p), 4),
        "frac_gt1": round(sum(1 for x in p if x > 1 + 1e-12) / len(p), 4),
    }
    if label:
        log("  K=%-4s n=%-6d mean=%.4f median=%.4f p95=%.4f "
            "min=%.4f max=%.4f tie=%.4f" %
            (label, s["n"], s["mean"], s["median"], s["p95"],
             s["min"], s["max"], s["frac_tie"]))
    return s


def run_arm(args, K, fh):
    lad = refine_ladder(K)
    patch = apply_ladder(lad)
    log("arm K=%d  ladder=%s" % (K, json.dumps(patch["ladder"])[:160]), fh)
    log("  range [%.2f, %.2f] preserved -> Theorem 1 bound = %.4f (unchanged)"
        % (lad[-1][1], lad[0][1], patch["theorem1_bound"]), fh)

    geo = base_geometry(seed=2024, side=2200.0)
    configs = enumerate_configs(budget=None)
    assert len(configs) == 15207, "config space drifted (%d)" % len(configs)
    if args.limit:
        configs = configs[:args.limit]

    t0 = time.time()
    records = []
    if args.workers > 1:
        from multiprocessing import Pool
        payload = [(t, geo, K, lad) for t in configs]
        with Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker, payload,
                                                      chunksize=32), 1):
                records.append(r)
                if i % 2000 == 0:
                    log("    %d/%d  elapsed=%.0fs" % (i, len(configs),
                                                      time.time() - t0), fh)
    else:
        for i, t in enumerate(configs, 1):
            core_solver.S2_T[:] = [tuple(p) for p in lad]
            from core_solver_exact import sync_stairs
            sync_stairs()
            records.append(solve_one(t, geo, K))
            if i % 2000 == 0:
                log("    %d/%d  elapsed=%.0fs" % (i, len(configs),
                                                  time.time() - t0), fh)

    records = [r for r in records if r]
    out = results_path(K)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    summary = summarise(records, str(K))
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"K": K, "patch": patch,
                   "parameters": dict(seed=2024, side_m=2200.0,
                                      n_communities=10, workers=args.workers,
                                      limit=args.limit),
                   "summary": summary, "elapsed_s": round(time.time() - t0, 1),
                   "results": records}, f)
    log("  -> %s  (%.0fs)" % (out, time.time() - t0), fh)
    return summary


def results_path(K):
    return os.path.join(constants.PKG_ROOT, "results", "sensitivity",
                        "ladder_K%d_per_config.json" % int(K))


def run_control(args, fh):
    """Re-run the PUBLISHED ladder and compare per-config against the archive."""
    lad = refine_ladder(5)
    apply_ladder(lad)
    geo = base_geometry(seed=2024, side=2200.0)
    configs = enumerate_configs(budget=None)[:args.control]
    arch = json.load(open(ARCHIVE, encoding="utf-8"))["results"]
    idx = {tuple(r["config"]): r for r in arch}

    log("control: %d configs, published ladder, vs %s"
        % (len(configs), os.path.basename(ARCHIVE)), fh)
    worst, ncmp = 0.0, 0
    for i, t in enumerate(configs, 1):
        r = solve_one(t, geo, 5)
        a = idx.get(tuple(t))
        if r and a:
            worst = max(worst, abs(r["PoA"] - a["PoA"]))
            ncmp += 1
        if i % 20 == 0:
            log("  %d/%d  max|dPoA|=%.3e" % (i, len(configs), worst), fh)
    log("control: n=%d  max|dPoA| = %.3e" % (ncmp, worst), fh)
    if worst < 1e-9:
        log("=> CONTROL EXACT (archive is bit-for-bit reproducible)", fh)
    elif worst < 1e-6:
        log("=> CONTROL OK at 1e-6 (solver-level noise only)", fh)
    else:
        log("=> CONTROL DRIFT: process-vs-archive differs (%.3e). " % worst +
            "Arms must be compared WITHIN this run, not against the archive.", fh)
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ladder", type=int, default=None,
                    help="number of S2 levels (5 = published control)")
    ap.add_argument("--all", action="store_true",
                    help="run K = 5, 10, 20, 50, 100 sequentially")
    ap.add_argument("--control", type=int, default=0,
                    help="control check: re-run N configs at K=5 vs archive")
    ap.add_argument("--limit", type=int, default=0, help="truncate config space")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(os.path.join(constants.PKG_ROOT, "results", "logs"),
                exist_ok=True)
    tag = ("control%d" % args.control) if args.control else \
          ("all" if args.all else "K%d" % args.ladder)
    fh = open(os.path.join(constants.PKG_ROOT, "results", "logs",
                           "ladder_%s.log" % tag), "w", encoding="utf-8")
    try:
        if args.control:
            run_control(args, fh)
        elif args.all:
            for K in (5, 10, 20, 50, 100):
                run_arm(args, K, fh)
        elif args.ladder:
            run_arm(args, args.ladder, fh)
        else:
            ap.error("give --ladder K, --all, or --control N")
    finally:
        fh.close()


if __name__ == "__main__":
    main()
