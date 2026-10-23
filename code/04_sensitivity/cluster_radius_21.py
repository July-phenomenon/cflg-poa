#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cluster-radius factor sensitivity for the 21-city study (paper §5.5).

The peak-detection radius ``r_c`` of a city is multiplied by a *factor* before
the neighbourhood graph is built, so the factor rescales how concentrated each
provider's catchment is.  The paper sweeps three factors:

    factor 0.8   tighter clustering  -> more concentrated catchments
    factor 1.0   as calibrated        -> the primary run
    factor 1.2   looser clustering    -> more dispersed catchments

Each cell re-runs all 21 cities with the **exact** single-MILP Phase 2 and the
coverage-locked Nash specification, i.e. exactly the primary 21-city run with a
different factor.  Results are checkpointed per city, so an interrupted sweep
resumes where it stopped (see ``exp1_21cities.py``).

This is a thin driver around ``exp1_21cities.py``; it does not re-implement any
model.  It writes one JSON per factor into ``results/main/`` and prints the
summary table quoted in the manuscript.

Usage
-----
    python code/04_sensitivity/cluster_radius_21.py
    python code/04_sensitivity/cluster_radius_21.py --factors 0.8 1.2 --time-limit 180

Note: ``cluster_radius.py`` in the same directory is an unrelated legacy
scale-feasibility probe (it uses the pre-2026 budget of 120 万元 per station and
reads a local competition directory that is not redistributed).  It produces no
paper number and is kept only for historical reference.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get(
    "CFLG_ROOT",
    os.path.abspath(os.path.join(HERE, "..", "..")))
EXP1 = os.path.join(ROOT, "code", "03_experiments", "exp1_21cities.py")
OUTDIR = os.path.join(ROOT, "results", "main")

DEFAULT_FACTORS = (0.8, 1.0, 1.2)
# Universal bound of Proposition 1 (constant, theta-independent): 1/0.6.
UNIVERSAL_BOUND = 1.0 / 0.6


def cell_path(factor):
    name = "21cities_exact_locked.json" if abs(factor - 1.0) < 1e-12 \
        else "21cities_cf%g_exact_locked.json" % factor
    return os.path.join(OUTDIR, name)


def run_cell(factor, budget, time_limit, nash_mode, resume):
    out = cell_path(factor)
    cmd = [sys.executable, EXP1,
           "--exact-phase2",
           "--nash-mode", nash_mode,
           "--budget-per-station", str(budget),
           "--cluster-factor", str(factor),
           "--output", out]
    if time_limit:
        cmd += ["--phase2-time-limit", str(time_limit)]
    if not resume:
        cmd += ["--no-resume"]
    print("\n=== cluster factor %g ===\n$ %s" % (factor, " ".join(cmd)))
    sys.stdout.flush()
    subprocess.check_call(cmd)
    return json.load(open(out, encoding="utf-8"))


def summarise(cells):
    """Return the per-cell statistics the manuscript quotes."""
    rows = []
    for factor, doc in cells:
        res = doc["results"]
        poa = np.array([v["PoA"] for v in res.values()])
        nonopt = [k for k, v in res.items() if v.get("milp_status") != "Optimal"]
        rows.append(dict(factor=factor, n=len(res),
                         mean=float(poa.mean()), median=float(np.median(poa)),
                         max=float(poa.max()), min=float(poa.min()),
                         nonoptimal=nonopt,
                         violations=int((poa > UNIVERSAL_BOUND + 1e-9).sum())))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--factors", type=float, nargs="+", default=list(DEFAULT_FACTORS))
    ap.add_argument("--budget-per-station", type=float, default=36.0)
    ap.add_argument("--nash-mode", default="locked", choices=["locked", "classic"])
    ap.add_argument("--time-limit", type=float, default=180.0,
                    help="per-city Phase-2 MILP time limit in seconds (0 = unlimited)")
    ap.add_argument("--no-resume", action="store_true",
                    help="recompute every city even if a checkpoint exists")
    ap.add_argument("--summary-only", action="store_true",
                    help="skip the runs and only re-summarise the stored JSON files")
    args = ap.parse_args()

    cells = []
    if args.summary_only:
        for f in args.factors:
            p = cell_path(f)
            if not os.path.exists(p):
                print("missing", os.path.relpath(p, ROOT))
                continue
            cells.append((f, json.load(open(p, encoding="utf-8"))))
    else:
        for f in args.factors:
            cells.append((f, run_cell(f, args.budget_per_station, args.time_limit,
                                      args.nash_mode, not args.no_resume)))

    rows = summarise(cells)
    print("\n%-8s %5s %9s %9s %9s %9s %10s %9s" %
          ("factor", "n", "mean", "median", "max", "min", "violations", "non-opt"))
    for r in rows:
        print("%-8g %5d %9.4f %9.4f %9.4f %9.4f %10d %9s" %
              (r["factor"], r["n"], r["mean"], r["median"], r["max"], r["min"],
               r["violations"], ",".join(r["nonoptimal"]) or "-"))
    if rows:
        means = [r["mean"] for r in rows]
        worst = max(r["max"] for r in rows)
        print("\nband of mean PoA: %.4f - %.4f (width %.4f); worst single-city PoA %.4f "
              "vs bound %.4f" % (min(means), max(means), max(means) - min(means),
                                 worst, UNIVERSAL_BOUND))
    out = os.path.join(OUTDIR, "cluster_radius_sensitivity_summary.json")
    json.dump(dict(budget_per_station_wan=args.budget_per_station,
                   nash_mode=args.nash_mode,
                   phase2_time_limit_s=args.time_limit,
                   universal_bound=UNIVERSAL_BOUND,
                   cells=rows),
              open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("summary ->", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
