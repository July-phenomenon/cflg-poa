# -*- coding: utf-8 -*-
"""Write the two figure inputs consumed by make_figures_paper.py.

`make_figures_paper.py` (the generator of the submitted figures) reads

    results/tables/_exact_src/Table5_2_21cities_PoA.csv        city, PoA
    results/tables/_exact_src/Table7_1_subsidy_comparison.csv  city, PoA_base,
                                                               PoA_subsidized_consistent

from the EXACT runs.  Before 2026-10-23 these two files were produced ad hoc, so
the figure pipeline could not be replayed from a clean checkout.  This script
closes that gap: it rebuilds both files from

    results/main/21cities_exact_locked.json
    results/main/_subsidy_exact/subsidy_exact.json

Run:  python code/05_figures_tables/export_exact_figure_inputs.py
Then: python code/05_figures_tables/make_figures_paper.py
"""
import csv
import json
import os

ROOT = os.environ.get(
    "CFLG_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
NUM = os.path.join(ROOT, "results", "main")
OUT = os.path.join(ROOT, "results", "tables", "_exact_src")

C21 = os.path.join(NUM, "21cities_exact_locked.json")
SUB = os.path.join(NUM, "_subsidy_exact", "subsidy_exact.json")


def main():
    os.makedirs(OUT, exist_ok=True)

    d21 = json.load(open(C21, encoding="utf-8"))
    assert d21["nash_mode"] == "locked", d21["nash_mode"]
    rows = sorted(d21["results"].items())
    p1 = os.path.join(OUT, "Table5_2_21cities_PoA.csv")
    with open(p1, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["city", "PoA"])
        for city, v in rows:
            w.writerow([city, "%.6f" % float(v["PoA"])])
    print("written ->", os.path.relpath(p1, ROOT), "(%d cities)" % len(rows))

    dsub = json.load(open(SUB, encoding="utf-8"))
    sres = dsub["results"]
    srows = sorted(sres.items()) if isinstance(sres, dict) else [(r["city"], r) for r in sres]
    p2 = os.path.join(OUT, "Table7_1_subsidy_comparison.csv")
    with open(p2, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["city", "PoA_base", "PoA_subsidized_consistent"])
        for city, v in srows:
            w.writerow([city, "%.6f" % float(v["PoA_base"]),
                        "%.6f" % float(v["PoA_sub_consistent"])])
    print("written ->", os.path.relpath(p2, ROOT), "(%d cities)" % len(srows))


if __name__ == "__main__":
    main()
