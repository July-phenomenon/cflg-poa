# -*- coding: utf-8 -*-
"""Global constants shared by every experiment.

All values are traceable to the competition specification (赛题附件 3 / 附件 5)
or to the fixed experimental design; nothing here is a free parameter.
"""
import os

# ---------------------------------------------------------------------------
# Objective weights (赛题附件 5):  F2 = W1*A1 + W2*A2 + W3*S3*cov
# ---------------------------------------------------------------------------
W1, W2, W3 = 0.2, 0.3, 0.5
S3 = 0.60                      # baseline S3 (constant satisfaction term)

# S1: distance -> satisfaction (附件 5 step function)
S1_T = [(300.0, 1.0), (500.0, 0.9), (650.0, 0.75), (1000.0, 0.6)]
# S2: utilisation -> satisfaction (附件 5 step function)
S2_T = [(0.6, 1.0), (0.75, 0.93), (0.85, 0.85), (0.95, 0.72), (1.0, 0.6)]

# Station capacity by size tier (人次/日), index 0 = "no station"
CAPACITY = (0.0, 1000.0, 2000.0, 3000.0)
# Construction cost by size tier (万元), read from 附件 3 when available
BUILD_COST_FALLBACK = (0.0, 18.0, 32.0, 45.0)

# ---------------------------------------------------------------------------
# Experimental design constants
# ---------------------------------------------------------------------------
BUDGET_PER_STATION_WAN = 36.0   # budget per station (万元)
DEMAND_OVER_CAPACITY = 1.15     # target total demand / total capacity
NASH_MAX_ITER = 10              # best-response iterations
NASH_CONV_TOL = 0.01            # convergence tolerance on S2

# Random seeds: fixed everywhere. `seed_for` (core solver) derives a stable
# per-city geometry seed; the enumeration uses SEED_ENUM.
SEED_ENUM = 2024

# ---------------------------------------------------------------------------
# Package layout
# ---------------------------------------------------------------------------
PKG_ROOT = os.environ.get(
    "CFLG_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..")))


def results_dir(*parts):
    return os.path.join(PKG_ROOT, "results", *parts)


def data_dir(*parts):
    return os.path.join(PKG_ROOT, "data", *parts)
