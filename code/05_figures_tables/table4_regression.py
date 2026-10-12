# -*- coding: utf-8 -*-
"""Table 4 - regression of PoA on structural predictors (section 5.4)

    python code/05_figures_tables/table4_regression
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, ".."))):
    if p not in sys.path:
        sys.path.insert(0, p)

if __name__ == "__main__":
    runpy.run_path(os.path.abspath(os.path.join(HERE, "..", "03_experiments/exp1_regression.py")),
                   run_name="__main__")
