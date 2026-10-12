# -*- coding: utf-8 -*-
"""Table 2 - instance parameters for the 21 cities (section 5.1)

Thin entry point required by the reproduction checklist.  The table body is
produced by ``make_tables.py``; this wrapper runs it and reports the artefact that
corresponds to this table.

    python code/05_figures_tables/table2_city_params
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

if __name__ == "__main__":
    runpy.run_path(os.path.join(HERE, "make_tables.py"), run_name="__main__")
    print("written ->", os.path.join("results", "tables", "Table5_1_21cities_params.csv"))
