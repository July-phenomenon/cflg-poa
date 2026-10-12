# -*- coding: utf-8 -*-
"""Figure 1 - sequential PoA across the 21 real cities (lollipop)

Thin entry point required by the reproduction checklist.  The drawing logic
lives in ``make_figures.py``; this wrapper only selects one panel so that a
single figure can be regenerated on its own.

    python code/05_figures_tables/fig1_poa21
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from make_figures import fig1_poa21          # noqa: E402

if __name__ == "__main__":
    fig1_poa21()
    print("written ->", os.path.join("results", "figures", "figure1.pdf"))
