# -*- coding: utf-8 -*-
"""Figure 2 - PoA distribution over the full enumeration (histogram + KDE)

Thin entry point required by the reproduction checklist.  The drawing logic
lives in ``make_figures.py``; this wrapper only selects one panel so that a
single figure can be regenerated on its own.

    python code/05_figures_tables/fig2_poa_dist
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from make_figures import fig2_poa_dist          # noqa: E402

if __name__ == "__main__":
    fig2_poa_dist()
    print("written ->", os.path.join("results", "figures", "figure2.pdf"))
