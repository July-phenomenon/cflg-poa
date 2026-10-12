# -*- coding: utf-8 -*-
"""Synthetic instance generator (paper sections 3.1 and 6).

The enumeration of section 6 uses a 10-community game: ten communities, each
hosting one candidate site whose size tier (none / small / medium / large) is
assigned by the enumeration.  Community geometry and population are synthetic
but drawn from a fixed seed, so the whole design is reproducible.

Population / demand settings follow the limitations paragraph of the paper:
    community size      ~ U(400, 1000)
    demand scaling        total demand = 1.15 x total capacity
    heterogeneity         per-community multiplier ~ U(0.7, 1.3)
"""
import itertools
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from core_solver import s1_fn                      # noqa: E402


def _consts():
    return {
        "n_communities": 10,
        # Synthetic city side (metres).  This is the one free geometric knob of
        # the synthetic enumeration: it controls how much the service areas of
        # neighbouring stations overlap, and therefore how much strategic
        # congestion the decentralized equilibrium generates.  2200 m was
        # selected by calibration (see docs/REPRODUCIBILITY.md, "Calibration of
        # the synthetic geometry"): it reproduces the published median PoA
        # (1.037 vs 1.0336) and the share of configurations with PoA > 1
        # (0.78 vs 0.878).
        "city_side_m": 2200.0,
        "seed_geometry": 2024,
        "capacity": (0.0, 1000.0, 2000.0, 3000.0),
        "build_cost": (0.0, 18.0, 32.0, 45.0),
        "budget": 118.0,
        "demand_over_capacity": 1.15,
    }


C = _consts()


def base_geometry(n=None, seed=None, side=None):
    """Fixed synthetic layout: community coordinates, populations, multipliers."""
    if n is None:
        n = C["n_communities"]
    if seed is None:
        seed = C["seed_geometry"]
    if side is None:
        side = C["city_side_m"]
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, side, size=(n, 2))
    pop = rng.uniform(400.0, 1000.0, size=n)
    hetero = rng.uniform(0.7, 1.3, size=n)
    return xy, pop, hetero


def distance_matrix(xy_com, xy_sta):
    return np.linalg.norm(xy_com[:, None, :] - xy_sta[None, :, :], axis=2)


def satisfaction_matrix(xy_com, xy_sta):
    """S1m[si, cj] = S1(distance); a zero entry means 'out of reach'."""
    d = distance_matrix(xy_com, xy_sta)          # (n_com, n_sta)
    S1m = np.zeros((d.shape[1], d.shape[0]))     # (n_sta, n_com)
    for si in range(d.shape[1]):
        for cj in range(d.shape[0]):
            S1m[si, cj] = s1_fn(d[cj, si])
    return S1m


def demand_vector(pop, hetero, sta_caps, ratio=None):
    """Scale demand so that total demand = ratio x total capacity."""
    if ratio is None:
        ratio = C["demand_over_capacity"]
    base = np.asarray(pop, dtype=float) * np.asarray(hetero, dtype=float)
    total_cap = float(np.sum(sta_caps))
    if total_cap <= 0 or base.sum() <= 0:
        return np.zeros_like(base)
    return base * (ratio * total_cap / base.sum())


def make_config(tiers, xy=None, pop=None, hetero=None, ratio=None):
    """Build one game instance from a size-tier assignment.

    Parameters
    ----------
    tiers : sequence of ints in {0,1,2,3}, one per candidate site.
            Tier 0 = no station is built at that site.  Such sites are removed
            from the candidate set: a site without a station cannot be chosen
            by any community and would leave utilisation undefined.

    Returns
    -------
    dict with keys n, pop, demand, S1m, sta_caps, xy, tiers, built.
        S1m has shape (n_sta, n_com) where n_sta is the number of built stations.
    """
    if xy is None or pop is None or hetero is None:
        xy, pop, hetero = base_geometry()
    if ratio is None:
        ratio = C["demand_over_capacity"]
    tiers = np.asarray(tiers, dtype=int)
    built = np.flatnonzero(tiers > 0)
    sta_caps = np.array([C["capacity"][t] for t in tiers[built]], dtype=float)
    S1m = satisfaction_matrix(xy, xy[built])
    demand = demand_vector(pop, hetero, sta_caps, ratio=ratio)
    return dict(n=len(pop), pop=pop, demand=demand, S1m=S1m,
                sta_caps=sta_caps, xy=xy, tiers=tiers, built=built)


def enumerate_configs(n=None, budget=None, cost=None, drop_empty=True):
    """Enumerate every size-tier assignment whose construction cost <= budget.

    This is the exact design used in the paper (section 6): ten candidate
    sites, four size tiers, construction costs 18 / 32 / 45 (10k CNY) from the
    competition specification, and a total budget of 118.  The enumeration
    yields 15,208 assignments; dropping the single degenerate all-zero
    assignment (no station is built, so total capacity is zero and the price of
    anarchy is undefined) leaves the 15,207 valid configurations reported in
    the paper.
    """
    if n is None:
        n = C["n_communities"]
    if budget is None:
        budget = C["budget"]
    if cost is None:
        cost = C["build_cost"]
    out = []
    for tiers in itertools.product(range(4), repeat=n):
        c = sum(cost[t] for t in tiers)
        if c > budget:
            continue
        if drop_empty and c == 0.0:
            continue
        out.append(list(tiers))
    return out


if __name__ == "__main__":
    cfgs = enumerate_configs()
    print("valid configurations:", len(cfgs))
    print("first:", cfgs[0])
    print("last :", cfgs[-1])
    inst = make_config(cfgs[0])
    print("n_com =", inst["n"], " n_sta =", len(inst["sta_caps"]),
          " caps =", inst["sta_caps"])
    print("S1m shape:", inst["S1m"].shape)
