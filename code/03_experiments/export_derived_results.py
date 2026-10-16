# -*- coding: utf-8 -*-
"""Export the derived enumeration results as a publishable CSV.

The competition specification (station sizes, capacity and cost tiers,
satisfaction ladders) and the artefacts derived *directly* from it are not
redistributable.  The table produced here is a **processed** result: it
contains only one row per enumerated configuration with

    config_id, PoA, F2_MILP, F2_Nash

i.e. the two objectives that enter the Price of Anarchy and the ratio
itself.  No competition input parameter appears in the output.

Usage
-----
    python code/03_experiments/export_derived_results.py \
        --src  <path to poa_full_per_config_s2.json> \
        --dest results/enumeration_poa_results.csv

If ``--src`` is omitted the script looks for the enumeration output in the
usual locations and exits with a short instruction when it is absent (the
enumeration itself is regenerated with ``exp2_enumeration.py``).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils.io_utils import root, ensure_dir  # noqa: E402

CANDIDATE_SOURCES = [
    "results/main/poa_full_per_config_s2.json",
    "results/main/poa_full_per_config.json",
]

HEADER = ["config_id", "PoA", "F2_MILP", "F2_Nash"]


def load_records(src: str):
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("results", "configs", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
        raise SystemExit("Unrecognised JSON layout in %s" % src)
    if isinstance(data, list):
        return data
    raise SystemExit("Unrecognised JSON layout in %s" % src)


def is_degenerate(rec) -> bool:
    """The all-zero configuration has no Nash/MILP solution and no PoA."""
    cfg = rec.get("config")
    if isinstance(cfg, (list, tuple)):
        return all(v == 0 for v in cfg)
    return rec.get("PoA") is None


def build_rows(records):
    rows = []
    cid = 0
    for rec in records:
        if is_degenerate(rec):
            continue
        cid += 1
        rows.append([
            cid,
            round(float(rec["PoA"]), 6),
            round(float(rec["milp_F2"]), 6),
            round(float(rec["nash_F2"]), 6),
        ])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=None,
                    help="enumeration output JSON (poa_full_per_config_s2.json)")
    ap.add_argument("--dest", default=root("results", "enumeration_poa_results.csv"),
                    help="destination CSV path")
    args = ap.parse_args()

    src = args.src
    if src is None:
        for rel in CANDIDATE_SOURCES:
            if os.path.exists(root(rel)):
                src = root(rel)
                break
    if src is None or not os.path.exists(src):
        print("Enumeration output not found.")
        print("Regenerate it first:")
        print("  python code/03_experiments/exp2_enumeration.py --workers 8")
        print("then re-run:")
        print("  python code/03_experiments/export_derived_results.py "
              "--src <that file>")
        return 1

    rows = build_rows(load_records(src))
    ensure_dir(os.path.dirname(os.path.abspath(args.dest)))
    with open(args.dest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print("wrote %s (%d rows)" % (args.dest, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
