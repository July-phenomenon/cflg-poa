# -*- coding: utf-8 -*-
"""Table 7 - three-specification comparison (appendix A)

Thin entry point required by the reproduction checklist.  The table body is
produced by ``make_tables.py``; this wrapper runs it and reports the artefact that
corresponds to this table.

    python code/05_figures_tables/table7_three_specs
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
    print("written ->", os.path.join("results", "tables", "TableA_1_three_specs.csv"))
