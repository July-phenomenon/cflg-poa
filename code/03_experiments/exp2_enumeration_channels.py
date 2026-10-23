# -*- coding: utf-8 -*-
"""Optimized full enumeration WITH channel decomposition (geometry-unification run).

This is a single-pass variant of exp2_enumeration.py written for the P2 geometry
unification: both the MILP and Nash solves are called ONCE with return_details=True
so that F2 and the per-channel quantities (A1, A2, S2) are recovered in a single
pass.  The original exp2_enumeration.py solves each config twice (once plain, once
with details); this halves the wall-clock for the 15,207-config sweep.

It uses the SAME canonical geometry as the Appendix-C sensitivity sweep
(base_geometry(seed=2024, side=2200)) so that the primary enumeration and the
sensitivity sweep share one geometry (the source of the old 7.9% vs 28.4%
locked-share discrepancy is thereby removed).

Output: results/main/results_enumeration.json  (same schema as exp2_enumeration.py)
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
    """Single-pass solve; returns a record dict (or None if degenerate)."""
    inst = make_config(tiers, xy=geo[0], pop=geo[1], hetero=geo[2])
    n, pop, demand, S1m, caps = (inst["n"], inst["pop"], inst["demand"],
                                 inst["S1m"], inst["sta_caps"])

    # ONE Nash solve with details
    nash_F2, nash_F1, _cov, _choices, nash_S2 = run_nash(
        n, pop, demand, S1m, caps, return_details=True)
    # ONE MILP solve with details
    milp_F2, milp_F1, _mcov, _w, milp_S2 = run_milp_exact(
        n, pop, demand, S1m, caps, return_details=True)

    if nash_F2 <= 0:
        return None

    # --- channel decomposition of ΔF2 (§6.3) -------------------------------
    # F2 = W1*A1 + W2*A2 + W3*S3*cov ; recover A1 and A2 from the totals.
    d_cov = milp_F1 - nash_F1
    d_A2 = float(milp_S2.mean() - nash_S2.mean()) if len(milp_S2) else 0.0
    A1_n = (nash_F2 - W2 * float(nash_S2.mean()) - W3 * S3 * nash_F1) / W1
    A1_m = (milp_F2 - W2 * float(milp_S2.mean()) - W3 * S3 * milp_F1) / W1
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


def main(argv=None):
    ap = argparse.ArgumentParser(description="§6 full enumeration (single-pass, channels)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only run the first N configurations (0 = all)")
    ap.add_argument("--seed", type=int, default=2024,
                    help="geometry seed (canonical = 2024)")
    ap.add_argument("--side", type=float, default=2200.0,
                    help="synthetic city side in metres (canonical = 2200)")
    ap.add_argument("--out", default=None, help="output json path")
    args = ap.parse_args(argv)

    os.makedirs(results_dir("main"), exist_ok=True)
    os.makedirs(results_dir("logs"), exist_ok=True)
    out_path = args.out or results_dir("main", "results_enumeration.json")
    log_path = results_dir("logs", "exp2_channels_run.log")
    geom_meta = {"seed": args.seed, "side_m": args.side}

    geo = base_geometry(seed=args.seed, side=args.side)
    configs = enumerate_configs(budget=None)
    assert len(configs) == 15207, "config space drifted (%d)" % len(configs)
    if args.limit:
        configs = configs[:args.limit]

    fh = open(log_path, "w", encoding="utf-8")
    log("exp2(channels) enumeration: %d configurations  geo(seed=%d, side=%.0f)"
        % (len(configs), args.seed, args.side), fh)
    log("output -> %s" % out_path, fh)

    # --- resume support: skip configs already present in a matching output --
    records, t0 = [], time.time()
    done_keys = set()
    for _cand in (out_path, out_path + ".bak"):
        if not os.path.exists(_cand):
            continue
        try:
            _prev = json.load(open(_cand, encoding="utf-8"))
        except Exception as exc:
            log("skip unreadable %s (%r)" % (_cand, exc), fh)
            continue
        if _prev.get("geometry") != geom_meta:
            log("geometry mismatch in %s -> not resuming from it" % _cand, fh)
            continue
        done_keys = {tuple(r["config"]) for r in _prev.get("results", [])}
        records = list(_prev["results"])
        log("resumed from %s: %d configs already done" % (_cand, len(done_keys)), fh)
        break
    if not done_keys:
        log("no resumable output found -> fresh run", fh)

    def _flush():
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"geometry": geom_meta, "results": records}, f)
        os.replace(tmp, out_path)            # atomic on Windows

    n_skip = 0
    for i, tiers in enumerate(configs, 1):
        key = tuple(tiers)
        if key in done_keys:
            n_skip += 1
            continue
        try:
            rec = solve_one(tiers, geo)
        except Exception as exc:                       # keep the sweep alive
            log("config %d failed: %r" % (i, exc), fh)
            rec = None
        if rec is not None:
            records.append(rec)
        if i % 200 == 0 or i == len(configs):
            _poa = [r["PoA"] for r in records]
            log("%d/%d  done=%d  new=%d  mean PoA=%.4f  elapsed=%.0fs"
                % (i, len(configs), len(records), i - n_skip,
                   float(np.mean(_poa)) if _poa else 0.0,
                   time.time() - t0), fh)
            _flush()

    _flush()
    poa = np.array([r["PoA"] for r in records])
    log("done: %d records in %.1f s (skipped %d already-done)"
        % (len(records), time.time() - t0, n_skip), fh)
    if len(poa):
        log("mean PoA >= %.4f  median >= %.4f  max >= %.4f"
            % (poa.mean(), np.median(poa), poa.max()), fh)
    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
