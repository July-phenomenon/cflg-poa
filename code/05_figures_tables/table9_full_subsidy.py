# -*- coding: utf-8 -*-
"""Table 9 - full per-city subsidy comparison (appendix B)

    python code/05_figures_tables/table9_full_subsidy
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, ".."))):
    if p not in sys.path:
        sys.path.insert(0, p)

if __name__ == "__main__":
    runpy.run_path(os.path.abspath(os.path.join(HERE, "..", "03_experiments/exp3_policy_analysis.py")),
                   run_name="__main__")
