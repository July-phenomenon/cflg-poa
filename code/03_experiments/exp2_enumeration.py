# -*- coding: utf-8 -*-
"""Experiment 2 - full enumeration over all feasible station-size designs (§6).

For every size-tier assignment that satisfies the construction budget, this
script solves

    * the centralized two-phase MILP  (run_milp_exact)
    * the decentralized Nash equilibrium (run_nash)

on the SAME instance and records the measured price of anarchy,

    PoA = F2_MILP / F2_Nash  (a measured LOWER bound, see §3.3)

together with the per-channel quantities needed by the decomposition of §6.3
(coverage, assignment and congestion contributions to ΔF2).

Usage
-----
    python exp2_enumeration.py                 # full run (15,207 configs, slow)
    python exp2_enumeration.py --limit 200     # smoke test / partial run
    python exp2_enumeration.py --workers 8     # parallel over processes

Output
------
    results/main/results_enumeration.json
    results/logs/exp2_run.log
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

from constants import W1, W2, W3, S3, PKG_ROOT                 # noqa: E402
from instance_generator import (base_geometry, enumerate_configs,  # noqa: E402
                                make_config)
from core_solver import run_milp_exact, run_nash               # noqa: E402


def results_dir(*parts):
    return os.path.join(PKG_ROOT, "results", *parts)


def log(msg, fh=None):
    line = "[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def solve_one(tiers, geo):
    """Solve one configuration; returns a record dict (or None if degenerate)."""
    inst = make_config(tiers, xy=geo[0], pop=geo[1], hetero=geo[2])
    n, pop, demand, S1m, caps = (inst["n"], inst["pop"], inst["demand"],
                                 inst["S1m"], inst["sta_caps"])

    nash_F2, nash_F1, _cov = run_nash(n, pop, demand, S1m, caps)
    milp_F2, milp_F1, _mcov = run_milp_exact(n, pop, demand, S1m, caps)

    if nash_F2 <= 0:
        return None

    # --- channel decomposition of ΔF2 (§6.3) -------------------------------
    # F2 = W1*A1 + W2*A2 + W3*S3*cov ; recover A1 and A2 from the totals.
    # A2 is the utilisation-satisfaction term, tracked explicitly by the
    # solver as the weighted S2 average.
    nash_F2d, nash_F1d, _covd, _ch, nash_S2 = run_nash(
        n, pop, demand, S1m, caps, return_details=True)
    milp_F2d, milp_F1d, _mcovd, _w, milp_S2 = run_milp_exact(
        n, pop, demand, S1m, caps, return_details=True)

    d_cov = milp_F1d - nash_F1d
    d_A2 = float(milp_S2.mean() - nash_S2.mean()) if len(milp_S2) else 0.0
    A1_n = (nash_F2d - W2 * float(nash_S2.mean()) - W3 * S3 * nash_F1d) / W1
    A1_m = (milp_F2d - W2 * float(milp_S2.mean()) - W3 * S3 * milp_F1d) / W1
    d_A1 = A1_m - A1_n

    return {
        "config": list(map(int, tiers)),
        "nash_F2": float(nash_F2),
        "milp_F2": float(milp_F2),
        "PoA": float(milp_F2 / nash_F2),
        "nash_F1": float(nash_F1),
        "milp_F1": float(milp_F1),
        "nash_S2_weighted": float(np.mean(nash_S2)),
        "milp_S2_weighted": float(np.mean(milp_S2)),
        "ch_assignment": float(W1 * d_A1),
        "ch_congestion": float(W2 * d_A2),
        "ch_coverage": float(W3 * S3 * d_cov),
    }


def _worker(payload):
    """Top-level helper so that the sweep can be mapped across a process pool."""
    tiers, geo = payload
    try:
        return solve_one(tiers, geo)
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="§6 full enumeration")
    ap.add_argument("--limit", type=int, default=0,
                    help="only run the first N configurations (0 = all)")
    ap.add_argument("--budget", type=float, default=None,
                    help="construction budget; default = the published 118")
    ap.add_argument("--out", default=None, help="output json path")
    ap.add_argument("--seed", type=int, default=2024,
                    help="geometry seed (fixed; do not change for replication)")
    ap.add_argument("--side", type=float, default=None,
                    help="synthetic city side in metres (default = calibrated 2200)")
    ap.add_argument("--workers", type=int, default=1,
                    help="process pool size (8 is a good choice on 8+ cores)")
    args = ap.parse_args(argv)

    os.makedirs(results_dir("main"), exist_ok=True)
    os.makedirs(results_dir("logs"), exist_ok=True)
    out_path = args.out or results_dir("main", "results_enumeration.json")
    log_path = results_dir("logs", "exp2_run.log")

    geo = base_geometry(seed=args.seed, side=args.side)
    configs = enumerate_configs(budget=args.budget)
    if args.limit:
        configs = configs[:args.limit]

    fh = open(log_path, "a", encoding="utf-8")
    log("exp2 enumeration: %d configurations (budget filter applied)" % len(configs), fh)
    log("output -> %s" % out_path, fh)

    records, t0 = [], time.time()

    def _flush():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"results": records}, f)

    if args.workers > 1:
        from multiprocessing import Pool
        payload = [(t, geo) for t in configs]
        with Pool(args.workers) as pool:
            for i, rec in enumerate(pool.imap(_worker, payload, chunksize=8), 1):
                if rec is not None:
                    records.append(rec)
                if i % 200 == 0 or i == len(configs):
                    done = [r["PoA"] for r in records]
                    log("%d/%d  n=%d  mean PoA=%.4f  elapsed=%.0fs"
                        % (i, len(configs), len(done),
                           float(np.mean(done)) if done else 0.0,
                           time.time() - t0), fh)
                    _flush()
    else:
        for i, tiers in enumerate(configs, 1):
            try:
                rec = solve_one(tiers, geo)
            except Exception as exc:                       # keep the sweep alive
                log("config %d failed: %r" % (i, exc), fh)
                rec = None
            if rec is not None:
                records.append(rec)
            if i % 200 == 0 or i == len(configs):
                done = [r["PoA"] for r in records]
                log("%d/%d  n=%d  mean PoA=%.4f  elapsed=%.0fs"
                    % (i, len(configs), len(done),
                       float(np.mean(done)) if done else 0.0,
                       time.time() - t0), fh)
                _flush()

    _flush()

    poa = np.array([r["PoA"] for r in records])
    log("done: %d records in %.1f s" % (len(records), time.time() - t0), fh)
    if len(poa):
        log("mean PoA >= %.4f  median >= %.4f  max >= %.4f"
            % (poa.mean(), np.median(poa), poa.max()), fh)
    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
