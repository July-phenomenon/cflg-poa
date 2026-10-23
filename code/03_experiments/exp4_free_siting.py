# -*- coding: utf-8 -*-
"""Three-tier decentralization spectrum on the 21 real cities.

Motivation
----------
The main specification compares a planner who *both sites and allocates* against
a decentralized equilibrium that only *allocates* on a planner-chosen network.
A referee can reasonably object that this is partial decentralization: the siting
decision is never decentralized.  This script adds the missing tier.

Two rights can be separated:

    siting right  : who decides where stations go and how big they are
    access right  : who decides which station each community uses

and three tiers follow:

    T1  self-serving siting + Nash access      (full decentralization)
    T2  planner siting     + Nash access      (partial decentralization, main spec)
    T3  planner siting     + planner access   (centralized, the benchmark)

All three tiers share (i) the same budget, (ii) the same demand vector, and
(iii) the same satisfaction tables, so differences come only from the two rights.

T1 rule (explicit, because a siting *game* would need a cost-benefit payoff that
this specification does not define): communities are served in descending
population order and each builds, at its own location, the largest station its
remaining share of the budget allows.  Because the smallest station costs 18 wan
against a 36 wan per-station budget, uncoordinated self-provision fragments the
same money into many small stations instead of few large ones.  This is a
*rule-based reference point* for uncoordinated investment, not a game-theoretic
equilibrium, and is labelled as such in the paper.

Output: results/main/threetier_spectrum.json
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "02_core_solver")))

from core_solver import (seed_for, run_nash_locked, run_nash,  # noqa: E402
                         milp_phase1_coverage, BUILD_COST_DICT,
                         W1, W2, W3, BASELINE_S3, S1_T, S2_T)

CAPACITY = np.array([0, 1000.0, 2000.0, 3000.0])
PKG = os.path.abspath(os.path.join(_HERE, "..", ".."))
PARAMS = os.path.join(PKG, "data", "processed", "21city_parameters.json")


def s1_fn(d):
    for u, v in S1_T:
        if d <= u:
            return v
    return 0.0


def load_profiles(path):
    with open(path, encoding="utf-8") as f:
        recs = json.load(f)
    out = []
    for r in recs:
        if r.get("cluster_r_m") is None or int(r.get("n_stations", 0)) < 1:
            continue
        out.append((r["city"], int(r["n_communities"]),
                    float(r["cluster_r_m"]), int(r["n_stations"])))
    return out


def allocate_planner(n_sta, budget_wan, cost):
    """Paper's rule A: floor(2/3) large stations, remainder small (budget-feasible)."""
    n_large = max(1, int(n_sta * 2 / 3))
    n_small = n_sta - n_large
    while cost[3] * n_large + cost[1] * n_small > budget_wan and n_large > 1:
        n_large -= 1
        n_small += 1
    return [3] * n_large + [1] * n_small


def allocate_self_serving(pop, comm_locs, budget_wan, cost):
    """Uncoordinated self-provision: largest-population communities build first,
    each at its own location, buying the smallest station until the budget runs
    out (a community with a 36-wan share cannot afford a 45-wan large station,
    so the money fragments into small ones)."""
    order = np.argsort(-pop)
    sizes, sites = [], []
    left = float(budget_wan)
    for c in order:
        tier = None
        for t in (3, 2, 1):                       # prefer the biggest affordable
            if cost[t] <= left + 1e-9:
                tier = t
                break
        if tier is None:
            break
        left -= cost[tier]
        sizes.append(tier)
        sites.append(c)
    return sizes, sites


def build_s1m(comm_locs, site_locs, s1_fn):
    n_sta = len(site_locs)
    n = len(comm_locs)
    S1m = np.zeros((n_sta, n))
    for i in range(n_sta):
        d = np.linalg.norm(comm_locs - site_locs[i], axis=1)
        for j in range(n):
            S1m[i, j] = s1_fn(d[j])
    return S1m


def solve_nash(n, pop, demand, S1m, caps, mode):
    if mode == "locked":
        _m, F1 = milp_phase1_coverage(n, pop, demand, S1m, caps)
        return run_nash_locked(n, pop, demand, S1m, caps, F1_lock=F1 + 1e-9)
    return run_nash(n, pop, demand, S1m, caps)


def main():
    ap = argparse.ArgumentParser(description="three-tier decentralization spectrum")
    ap.add_argument("--budget-per-station", type=float, default=36.0)
    ap.add_argument("--output", default=os.path.join(PKG, "results", "main",
                                                     "threetier_spectrum.json"))
    args = ap.parse_args()

    cost = BUILD_COST_DICT
    profiles = load_profiles(PARAMS)
    print("cities: %d   budget/station: %.0f wan" % (len(profiles), args.budget_per_station))

    rows = {}
    for city, n, cluster_r, n_sta in profiles:
        rng = np.random.RandomState(seed_for(city))
        extent = cluster_r * 3.0
        comm_locs = rng.uniform(-extent / 2, extent / 2, size=(n, 2))
        fac_locs = rng.uniform(-cluster_r * 0.8, cluster_r * 0.8, size=(n_sta, 2))
        pop = rng.randint(400, 1000, size=n).astype(float)

        budget = args.budget_per_station * n_sta
        # ---- tier 3 / 2 network: planner siting (rule A) ----
        sizes_p = allocate_planner(n_sta, budget, cost)
        caps_p = np.array([CAPACITY[s] for s in sizes_p], dtype=float)
        S1m_p = build_s1m(comm_locs, fac_locs[:len(sizes_p)], s1_fn)
        # demand is fixed once, on the planner network, and reused by every tier
        target = caps_p.sum() * 1.15
        raw = rng.uniform(0.7, 1.3, size=n)
        demand = np.round(raw / raw.sum() * target).astype(np.int64)

        # ---- tier 1 network: self-serving siting ----
        sizes_s, sites_s = allocate_self_serving(pop, comm_locs, budget, cost)
        caps_s = np.array([CAPACITY[t] for t in sizes_s], dtype=float)
        S1m_s = build_s1m(comm_locs, comm_locs[sites_s], s1_fn)

        out = dict(n=int(n), n_sta=int(n_sta), budget_wan=float(budget),
                   planner=dict(n_sta=len(sizes_p), n_large=int(sum(1 for s in sizes_p if s == 3)),
                                capacity=float(caps_p.sum()),
                                cost=float(sum(cost[s] for s in sizes_p))),
                   self_serv=dict(n_sta=len(sizes_s), n_large=int(sum(1 for s in sizes_s if s == 3)),
                                  capacity=float(caps_s.sum()),
                                  cost=float(sum(cost[t] for t in sizes_s))))

        # T3: planner access (centralized two-phase benchmark, exact Phase 2 elsewhere)
        from core_solver_exact import run_two_phase_exact
        F2_c, F1_c, _ = run_two_phase_exact(n, pop, demand, S1m_p, caps_p)
        out["T3_central"] = dict(F2=float(F2_c), F1=float(F1_c))

        # T2: planner siting + Nash access
        F2_p, F1_p, _ = solve_nash(n, pop, demand, S1m_p, caps_p, "locked")
        out["T2_partial"] = dict(F2=float(F2_p), F1=float(F1_p))

        # T1: self-serving siting + Nash access (coverage locked on its own network)
        F2_s, F1_s, _ = solve_nash(n, pop, demand, S1m_s, caps_s, "locked")
        out["T1_self"] = dict(F2=float(F2_s), F1=float(F1_s))

        for k in ("T3_central", "T2_partial", "T1_self"):
            out[k]["cover_frac"] = float(out[k]["F1"])
        if F2_s > 1e-9:
            out["PoA_T2_over_T3"] = float(F2_c / F2_p) if F2_p > 1e-9 else None
            out["PoA_T1_over_T3"] = float(F2_c / F2_s)
        rows[city] = out
        print("%-10s n=%4d  T3_F2=%.4f  T2_F2=%.4f  T1_F2=%.4f  |  PoA_T2=%.4f  PoA_T1=%.4f"
              % (city, n, F2_c, F2_p, F2_s,
                 F2_c / F2_p if F2_p > 1e-9 else float("nan"),
                 F2_c / F2_s if F2_s > 1e-9 else float("nan")))

    def agg(key):
        v = np.array([r[key] for r in rows.values()])
        return dict(mean=float(v.mean()), median=float(np.median(v)),
                    min=float(v.min()), max=float(v.max()))

    summary = dict(
        n_cities=len(rows),
        budget_per_station_wan=args.budget_per_station,
        T1_self_serving=dict(F2=agg("T1_self") if False else
                             dict(mean=float(np.mean([r["T1_self"]["F2"] for r in rows.values()])),
                                  cover=float(np.mean([r["T1_self"]["cover_frac"] for r in rows.values()])))),
        T2_partial=dict(mean_F2=float(np.mean([r["T2_partial"]["F2"] for r in rows.values()])),
                        mean_cover=float(np.mean([r["T2_partial"]["cover_frac"] for r in rows.values()]))),
        T3_central=dict(mean_F2=float(np.mean([r["T3_central"]["F2"] for r in rows.values()])),
                        mean_cover=float(np.mean([r["T3_central"]["cover_frac"] for r in rows.values()]))),
        PoA_T2=agg("PoA_T2_over_T3"),
        PoA_T1=agg("PoA_T1_over_T3"),
        self_serving_network=dict(
            mean_n_sta=float(np.mean([r["self_serv"]["n_sta"] for r in rows.values()])),
            mean_capacity=float(np.mean([r["self_serv"]["capacity"] for r in rows.values()])),
            mean_large_share=float(np.mean([r["self_serv"]["n_large"] / max(1, r["self_serv"]["n_sta"])
                                            for r in rows.values()]))),
        planner_network=dict(
            mean_n_sta=float(np.mean([r["planner"]["n_sta"] for r in rows.values()])),
            mean_capacity=float(np.mean([r["planner"]["capacity"] for r in rows.values()]))),
        note=("T1 is a rule-based reference for uncoordinated self-provision "
              "(largest communities build first, smallest affordable station, "
              "no cost-benefit payoff), not a game-theoretic equilibrium."),
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dict(summary=summary, cities=rows), f, indent=2, ensure_ascii=False)
    print("\n=== SUMMARY ===")
    for k in ("PoA_T2", "PoA_T1"):
        print("%-8s mean=%.4f median=%.4f min=%.4f max=%.4f"
              % (k, summary[k]["mean"], summary[k]["median"],
                 summary[k]["min"], summary[k]["max"]))
    print("self-serving network: %.1f stations, capacity %.0f, large share %.0f%%"
          % (summary["self_serving_network"]["mean_n_sta"],
             summary["self_serving_network"]["mean_capacity"],
             100 * summary["self_serving_network"]["mean_large_share"]))
    print("planner    network: %.1f stations, capacity %.0f"
          % (summary["planner_network"]["mean_n_sta"], summary["planner_network"]["mean_capacity"]))
    print("saved ->", args.output)


if __name__ == "__main__":
    main()
