# -*- coding: utf-8 -*-
"""E1: coverage-band classification of the 21 city instances (paper Table "coverage bands").

Why this script exists
----------------------
Section~5 of the manuscript argues that the measured PoA is small *because*
coverage is low: when most of the elderly population sits on the censoring
floor, the planner and the residents want nearly the same thing, so the
efficiency gap collapses.  Until now that argument was carried by prose and by
the per-city table.  This script turns it into a formal classification:

    band A (low)    c^MILP <  20 %
    band B (medium) 20 % <= c^MILP < 35 %
    band C (high)   c^MILP >= 35 %

The 20 / 35 % cut points are the tercile-like breaks that keep every band
non-empty on n = 21 (4 / 11 / 6 cities) while remaining round numbers; the
result is insensitive to moving either cut by +/-2 pp (checked below and
reported in the log).

For each band the script reports the mean / median / min / max PoA, the mean
planner coverage, the mean coverage gap and the mean congestion gap.  The
monotone gradient across bands is the quantitative form of the mechanism.

Inputs
------
``results/main/21cities_exact_locked.json``  (primary 21-city run, locked spec)

Outputs
-------
``results/tables/T13_coverage_bands.csv``    machine readable
``results/tables/T13_coverage_bands.json``   same + cut-point sensitivity
``results/tables/T13_coverage_bands.tex``    LaTeX fragment quoted by the paper

Usage
-----
    python code/05_figures_tables/table_cov_bands.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get(
    "CFLG_ROOT", os.path.abspath(os.path.join(HERE, "..", "..")))
SRC = os.path.join(ROOT, "results", "main", "21cities_exact_locked.json")
OUTDIR = os.path.join(ROOT, "results", "tables")

# Cut points in coverage *share* (not per cent).
CUTS = (0.20, 0.35)
BAND_NAMES = ("Low (<20%)", "Medium (20-35%)", "High ($\\geq$35%)")


def load_cities(path):
    doc = json.load(open(path, encoding="utf-8"))
    res = doc.get("results", doc)
    rows = []
    for city, v in res.items():
        rows.append(dict(city=city,
                         cov=float(v["milp_cov"]),
                         nash_cov=float(v["nash_cov"]),
                         poa=float(v["PoA"]),
                         delta_cov=float(v["delta_cov"]),
                         delta_s2=float(v["delta_S2"]),
                         n_sta=int(v["n_stations"])))
    if len(rows) != 21:
        print(f"[warn] expected 21 cities, found {len(rows)}", file=sys.stderr)
    return sorted(rows, key=lambda r: r["cov"])


def band_of(cov, cuts=CUTS):
    lo, hi = cuts
    if cov < lo:
        return 0
    if cov < hi:
        return 1
    return 2


def summarise(rows, cuts=CUTS):
    out = []
    for b in range(3):
        sub = [r for r in rows if band_of(r["cov"], cuts) == b]
        if not sub:
            out.append(dict(band=b, name=BAND_NAMES[b], n=0))
            continue
        poa = np.array([r["poa"] for r in sub])
        out.append(dict(
            band=b,
            name=BAND_NAMES[b],
            n=len(sub),
            cities=[r["city"] for r in sub],
            poa_mean=float(poa.mean()),
            poa_median=float(np.median(poa)),
            poa_min=float(poa.min()),
            poa_max=float(poa.max()),
            cov_mean=float(np.mean([r["cov"] for r in sub])),
            delta_cov_mean=float(np.mean([r["delta_cov"] for r in sub])),
            delta_s2_mean=float(np.mean([r["delta_s2"] for r in sub])),
            n_cov_aligned=int(sum(abs(r["delta_cov"]) <= 1e-9 for r in sub)),
        ))
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = load_cities(SRC)
    cells = summarise(rows)

    print("21-city coverage-band classification (primary locked run)")
    print("%-18s %3s %9s %9s %9s %9s %9s %9s" %
          ("band", "n", "PoA mean", "median", "min", "max",
           "c^MILP", "d_cov"))
    for c in cells:
        print("%-18s %3d %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f" % (
            c["name"].replace("$\\geq$", ">="), c["n"], c["poa_mean"],
            c["poa_median"], c["poa_min"], c["poa_max"], c["cov_mean"],
            c["delta_cov_mean"]))

    # ---- cut-point sensitivity: move either break by +/-2 pp -------------
    sens = []
    for shift in (-0.02, 0.0, 0.02):
        cuts = (CUTS[0] + shift, CUTS[1] + shift)
        alt = summarise(rows, cuts)
        sens.append(dict(lo=round(cuts[0], 3), hi=round(cuts[1], 3),
                         n=[c["n"] for c in alt],
                         mean=[round(c["poa_mean"], 4) for c in alt]))
        print("cuts %.0f%%/%.0f%%: n=%s mean=%s" % (
            100 * cuts[0], 100 * cuts[1], [c["n"] for c in alt],
            [round(c["poa_mean"], 4) for c in alt]))

    means = [c["poa_mean"] for c in cells]
    mono = all(means[i] <= means[i + 1] for i in range(len(means) - 1))
    print("\nmonotone across bands:", mono)

    payload = dict(source=os.path.relpath(SRC, ROOT),
                   cuts_pct=[100 * c for c in CUTS],
                   bands=cells,
                   monotone=bool(mono),
                   cut_sensitivity=sens)
    with open(os.path.join(OUTDIR, "T13_coverage_bands.json"), "w",
              encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    import csv
    with open(os.path.join(OUTDIR, "T13_coverage_bands.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["band", "n_cities", "poa_mean", "poa_median", "poa_min",
                    "poa_max", "cov_mean", "delta_cov_mean", "delta_s2_mean",
                    "n_cov_aligned", "cities"])
        for c in cells:
            if c["n"] == 0:
                continue
            w.writerow([c["name"].replace("$\\geq$", ">="), c["n"],
                        round(c["poa_mean"], 4), round(c["poa_median"], 4),
                        round(c["poa_min"], 4), round(c["poa_max"], 4),
                        round(c["cov_mean"], 4), round(c["delta_cov_mean"], 4),
                        round(c["delta_s2_mean"], 4), c["n_cov_aligned"],
                        ";".join(c["cities"])])

    # ---- LaTeX fragment --------------------------------------------------
    lines = []
    for c in cells:
        if c["n"] == 0:
            continue
        lines.append(
            "%s & %d & %.4f & %.4f & [%.4f, %.4f] & %.1f & %.3f & %.4f & %d \\\\" % (
                c["name"], c["n"], c["poa_mean"], c["poa_median"],
                c["poa_min"], c["poa_max"], 100 * c["cov_mean"],
                100 * c["delta_cov_mean"], c["delta_s2_mean"], c["n_cov_aligned"]))
    with open(os.path.join(OUTDIR, "T13_coverage_bands.tex"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[saved] results/tables/T13_coverage_bands.{csv,json,tex}")


if __name__ == "__main__":
    main()
