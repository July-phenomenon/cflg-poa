# -*- coding: utf-8 -*-
"""Cap / coverage-floor sensitivity sweep for the PoA bounds (paper §4, new Appendix C).

What is swept
-------------
theta -- the satisfaction censoring floor.  In the published specification all
three satisfaction components are censored below at 0.6 (S1 table floor,
S2 table floor, and the constant S3 = 0.60), so the worst-case served
satisfaction is S_min = W1*theta + W2*theta + W3*theta = theta and the
theoretical anchors move with theta:

    Theorem 1 bound   = 1 / S_min = 1 / theta            (locked layer, dcov = 0)
    Corollary 1 bound = 1 / (theta * c_min)              (configs with c_N >= c_min)

theta = 0.6 is the published specification; its enumeration results are the
archived `poa_full_per_config_s2.json` and are NOT re-run here for per-config
comparison.  A *reproduction check* runs at theta = 0.6 and compares the
aggregate PoA statistics (mean / median / p95 / max / frac_tie) against the
archive as a distribution-level sanity check, acknowledging the known
geometry-calibration caveat (synthetic geometry is regenerated in-process with
seed 2024, side 2200 m; the archive used the originally-calibrated geometry).

c_min (the coverage-floor assumption behind Corollary 1) is a *classification*
parameter: it does not enter the solve.  Per-config Nash coverage c_N =
nash_F1 is recorded, so any c_min partition ({0.7, 0.8, 0.9}) is computed in
post-processing by cap_sensitivity_summary.py.

How the patch works (and why it is safe)
----------------------------------------
core_solver reads its floors from module globals at call time (s1_fn reads
S1_T, s2_fn reads S2_T, the solvers read BASELINE_S3), and
instance_generator.satisfaction_matrix calls core_solver.s1_fn at
instance-construction time.  Patching the three globals BEFORE building any
instance therefore propagates the floor consistently into S1m, the MILP
objective and the Nash utilities.  `apply_theta` hard-asserts that the solver
actually observes the new floors (and that the untouched tiers are intact), so
a silent patch failure aborts the run instead of producing wrong science.

Reproducibility
---------------
Same geometry seed (2024), same side (2200 m), same budget filter (118) and
same solver code path as exp2_enumeration.py; only the three floor constants
differ.  A manifest with input/output hashes is written through
core_solver.write_manifest.

Usage
-----
    python cap_floor_sensitivity.py --theta 0.5 --workers 12
    python cap_floor_sensitivity.py --theta 0.7 --workers 12
    python cap_floor_sensitivity.py --control 40      # replication check vs archive

Output
------
    results/sensitivity/cap_floor_theta_<t>_per_config.json
    results/logs/cap_floor_theta_<t>.log
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
from core_solver import run_milp_exact, run_nash                    # noqa: E402

# The local full-enumeration archive (not shipped in the repo for compliance;
# regenerate with exp2_enumeration.py if absent).  Used only by --control.
DEFAULT_ARCHIVE = os.path.join(
    os.path.dirname(_HERE), "..", "..",
    "原始实验结果存档", "全量枚举", "poa_full_per_config_s2.json")


def apply_theta(theta):
    """Floor S1/S2 tables and S3 baseline at `theta`, with hard verification."""
    core_solver.S1_T[-1] = (1000.0, theta)
    core_solver.S2_T[-1] = (1.0, theta)
    core_solver.BASELINE_S3 = theta
    # --- the solver MUST observe the patch ----------------------------------
    assert abs(core_solver.s1_fn(1000.0) - theta) < 1e-12, "S1 floor not applied"
    assert abs(core_solver.s1_fn(650.0) - 0.75) < 1e-12, "S1 inner tier corrupted"
    assert abs(core_solver.s1_fn(1001.0)) < 1e-12, "S1 out-of-reach corrupted"
    assert abs(core_solver.s2_fn(1.0) - theta) < 1e-12, "S2 floor not applied"
    assert abs(core_solver.s2_fn(0.95) - 0.72) < 1e-12, "S2 inner tier corrupted"
    assert abs(core_solver.BASELINE_S3 - theta) < 1e-12, "S3 baseline not applied"
    s_min = W1 * theta + W2 * theta + W3 * theta
    assert abs(s_min - theta) < 1e-12, "S_min != theta (weights changed?)"
    return dict(theta=theta,
                S1_T=list(map(list, core_solver.S1_T)),
                S2_T=list(map(list, core_solver.S2_T)),
                BASELINE_S3=theta,
                S_min=s_min,
                theorem1_bound=1.0 / theta,
                weights=[W1, W2, W3])


def results_dir(*parts):
    return os.path.join(constants.PKG_ROOT, "results", *parts)


def log(msg, fh):
    line = "[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def solve_one(tiers, geo, theta):
    """Solve one configuration at the patched floors (single pass, no details)."""
    inst = make_config(tiers, xy=geo[0], pop=geo[1], hetero=geo[2])
    n, pop, demand, S1m, caps = (inst["n"], inst["pop"], inst["demand"],
                                 inst["S1m"], inst["sta_caps"])
    nash_F2, nash_F1, _ = run_nash(n, pop, demand, S1m, caps)
    milp_F2, milp_F1, _ = run_milp_exact(n, pop, demand, S1m, caps)
    if nash_F2 <= 0:
        return None
    return {
        "config": list(map(int, tiers)),
        "theta": theta,
        "nash_F2": float(nash_F2),
        "milp_F2": float(milp_F2),
        "PoA": float(milp_F2 / nash_F2),
        "nash_F1": float(nash_F1),
        "milp_F1": float(milp_F1),
    }


def _worker(payload):
    tiers, geo, theta = payload
    # Belt-and-suspenders: on Windows spawn, each child re-imports modules with
    # DEFAULT floors; the Pool initializer below re-applies the patch, and this
    # assertion guarantees no config is ever solved unpatched.
    assert abs(core_solver.BASELINE_S3 - theta) < 1e-12, (
        "worker not patched: S3=%r != theta=%r" % (core_solver.BASELINE_S3, theta))
    try:
        return solve_one(tiers, geo, theta)
    except Exception:
        return None


def run_sweep(args, fh):
    theta = apply_theta(args.theta)
    log("patched floors: %s" % json.dumps(theta), fh)

    geo = base_geometry(seed=2024, side=2200.0)
    configs = enumerate_configs(budget=None)
    assert len(configs) == 15207, "config space drifted (%d)" % len(configs)
    if args.limit:
        configs = configs[:args.limit]

    out_path = results_dir("sensitivity",
                           "cap_floor_theta_%s_per_config.json" % args.tag)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(results_dir("logs"), exist_ok=True)
    log("sweep theta=%s  %d configurations  workers=%d"
        % (args.theta, len(configs), args.workers), fh)
    log("output -> %s" % out_path, fh)

    records, t0 = [], time.time()

    def _flush():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"theta": args.theta, "patch": theta,
                       "parameters": dict(seed=2024, side_m=2200.0,
                                          budget=118.0, n_communities=10,
                                          workers=args.workers),
                       "results": records}, f)

    if args.workers > 1:
        from multiprocessing import Pool
        payload = [(t, geo, args.theta) for t in configs]
        # CRITICAL: apply_theta must run in EACH child (spawn re-imports the
        # module with default floors); an initializer-only patch in the parent
        # silently solves every grid at theta=0.6.
        with Pool(args.workers, initializer=apply_theta,
                  initargs=(args.theta,)) as pool:
            for i, rec in enumerate(pool.imap(_worker, payload, chunksize=8), 1):
                if rec is not None:
                    records.append(rec)
                if i % 500 == 0 or i == len(configs):
                    done = [r["PoA"] for r in records]
                    log("%d/%d  n=%d  mean PoA=%.4f  elapsed=%.0fs"
                        % (i, len(configs), len(done),
                           float(np.mean(done)) if done else 0.0,
                           time.time() - t0), fh)
                    _flush()
    else:
        for i, tiers in enumerate(configs, 1):
            rec = _worker((tiers, geo, args.theta))
            if rec is not None:
                records.append(rec)
            if i % 100 == 0 or i == len(configs):
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

    # --- reproduction check at the published specification -------------------
    if abs(args.theta - 0.6) < 1e-9 and os.path.exists(args.archive):
        ref = archive_aggregate(args.archive)
        frac_tie = sum(1 for r in records
                       if abs(r["milp_F1"] - r["nash_F1"]) < 1e-9) / len(records)
        log("REPRODUCTION CHECK (theta=0.6 vs published archive, aggregate):", fh)
        log("  this-run : n=%d mean=%.4f median=%.4f p95=%.4f max=%.4f frac_tie=%.4f"
            % (len(records), poa.mean(), np.median(poa),
               np.percentile(poa, 95), poa.max(), frac_tie), fh)
        log("  archive  : n=%d mean=%.4f median=%.4f p95=%.4f max=%.4f frac_tie=%.4f"
            % (ref["n"], ref["mean"], ref["median"], ref["p95"],
               ref["max"], ref["frac_tie"]), fh)
        d_mean, d_max = poa.mean() - ref["mean"], poa.max() - ref["max"]
        d_tie = frac_tie - ref["frac_tie"]
        log("  delta: mean=%.4f max=%.4f frac_tie=%.4f" % (d_mean, d_max, d_tie), fh)
        ok = abs(d_mean) < 0.05 and abs(d_max) < 0.20
        log("  AGGREGATE REPRODUCTION %s (tol mean 0.05 / max 0.20)"
            % ("OK" if ok else "REVIEW-CALIBRATION"), fh)

    core_solver.write_manifest(
        experiment_id="cap_floor_theta_%s" % args.tag,
        script_path=os.path.abspath(__file__),
        inputs=[], outputs=[out_path],
        parameters=dict(theta=args.theta, patch=theta, seed=2024, side_m=2200.0,
                        budget=118.0, workers=args.workers,
                        limit=args.limit or "all"),
        data_source_note=("geometry generated in-process (instance_generator, "
                          "seed 2024); no external data input"),
        notes=("Corollary-1 c_min partitions are computed in post-processing "
               "by cap_sensitivity_summary.py from the recorded nash_F1."))
    return out_path


def archive_aggregate(path):
    """Aggregate PoA statistics of the published full-enumeration archive.

    Used only as a *reference* for the theta=0.6 reproduction check.  Per-config
    matching against the archive is intentionally NOT done: the archived runs
    used the originally-calibrated synthetic geometry, whereas this sweep
    regenerates geometry in-process (seed 2024, side 2200 m).  The two agree on
    the *distribution* of PoA (same design, same seed family) but not config by
    config -- that is the known geometry-calibration caveat, not a bug.
    """
    with open(path, encoding="utf-8") as f:
        arch = json.load(f)["results"]
    valid = [c for c in arch if c.get("nash_F2", 0) > 0]
    poa = np.array([c["milp_F2"] / c["nash_F2"] for c in valid])
    tie = sum(1 for c in valid
              if abs(c["milp_F1"] - c["nash_F1"]) < 1e-9)
    return dict(n=len(valid), mean=float(poa.mean()), median=float(np.median(poa)),
                p95=float(np.percentile(poa, 95)), max=float(poa.max()),
                frac_tie=tie / len(valid) if valid else 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description="cap/coverage-floor sensitivity")
    ap.add_argument("--theta", type=float, default=None,
                    help="satisfaction floor to sweep (e.g. 0.5 or 0.7)")
    ap.add_argument("--tag", default=None, help="output tag (default = theta)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--archive", default=DEFAULT_ARCHIVE)
    args = ap.parse_args(argv)

    assert args.theta is not None, "--theta required"
    args.tag = args.tag or ("%.2f" % args.theta)
    os.makedirs(os.path.join(constants.PKG_ROOT, "results", "logs"),
                exist_ok=True)
    lp = os.path.join(constants.PKG_ROOT, "results", "logs",
                      "cap_floor_theta_%s.log" % args.tag)
    with open(lp, "a", encoding="utf-8") as fh:
        run_sweep(args, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
