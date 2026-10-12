# -*- coding: utf-8 -*-
"""Table 8 - channel contributions to the efficiency gap (appendix A)

    python code/05_figures_tables/table8_channels
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, ".."))):
    if p not in sys.path:
        sys.path.insert(0, p)

if __name__ == "__main__":
    runpy.run_path(os.path.abspath(os.path.join(HERE, "..", "03_experiments/exp2_channel_decomp.py")),
                   run_name="__main__")
