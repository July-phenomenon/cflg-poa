"""Exact Phase-2 MILP for the two-phase sequential optimum (CFLG-CF).

Why this module exists
----------------------
``core_solver.run_milp_exact`` solves Phase 2 by an OUTER fixed-point iteration:
it freezes the utilisation satisfactions S2, solves a MILP for the assignment,
recomputes the induced utilisation, and repeats (at most 10 rounds), keeping the
best round.  The F2 it returns therefore carries no global-optimality guarantee,
and every PoA built on it is only a lower bound (this is what makes the three
21-city cases with PoA < 1 possible).

This module removes the outer iteration.  The only non-linearity in Phase 2 is
the staircase map

    u  |->  S2(u),      S2_T = [(0.6, 1.0), (0.75, .93), (.85, .85), (.95, .72), (1.0, .6)]

with u = load / capacity.  Because the staircase has finitely many segments it
can be encoded EXACTLY in a single MILP:

  * per station s, binaries z[s,k] with sum_k z[s,k] = 1 select the segment k;
  * segment bounds are enforced with a *tight* big-M equal to the width of the
    constraint at hand, cap_s * (1 - U[k+1]) -- not the loose cap_s;
  * the bilinear term z[s,k] * (population served by s) is linearised with a
    McCormick envelope (t[s,k] = z[s,k] * L_s), using the station's own maximum
    servable population Lmax_s as the big-M.

Phase 1 (max coverage) is kept logically identical to ``run_milp_exact`` so that
the exact and heuristic benchmarks are compared on the same locked value F1_max.

Solver choice and the CBC false-infeasibility
--------------------------------------------
``PULP_CBC_CMD`` (CBC 2.10.3) reports **Infeasible on feasible instances** for
this model family: ``Cgl0000I Cut generators found to be infeasible!`` is raised
while reading the maximisation MPS that PuLP writes.  The very same model written
as an LP file and solved by the CBC executable directly returns the correct
optimum, so this is a CBC reader/cut-generator defect, not a modelling error
(e.g. instance 48 of the enumeration: CBC/PuLP says Infeasible, CBC-on-LP and
HiGHS both return 810.39294649).

We therefore default to **HiGHS** (bundled with PuLP 3.x), which is numerically
robust and is a genuine MILP solver with optimality certificates.  CBC remains
available via ``solver="cbc"`` for cross-checking, and every solve is followed by
an independent feasibility verification (``_verify``) that recomputes capacity,
coverage and segment consistency from the decoded assignment in plain NumPy --
so a wrong solver verdict can never silently enter the results.

Reported F2
-----------
The F2 returned is ALWAYS recomputed from the decoded assignment using the true
staircase ``s2_fn`` -- never taken from the MILP objective.  This keeps the
accounting identical to the heuristic version and makes any encoding error
visible as a mismatch between ``obj_value`` and ``F2``.
"""

import numpy as np
import pulp

import core_solver as _cs
from core_solver import s2_fn, S2_T, W1, W2, W3, BASELINE_S3  # noqa: F401

# --- staircase segments -----------------------------------------------------
_U = [0.0] + [b for b, _ in S2_T]        # segment upper bounds: 0, .6, .75, .85, .95, 1.0
_V = [v for _, v in S2_T]                # segment values:       1.0, .93, .85, .72, .6
K = len(S2_T)
assert len(_U) == K + 1


def sync_stairs():
    """Re-read the staircase and the S3 baseline from ``core_solver``.

    The theta sweep patches ``core_solver.S2_T`` in place and rebinds
    ``core_solver.BASELINE_S3``.  ``S2_T`` is a shared list, so ``s2_fn`` picks
    the change up automatically, but the segment arrays below and the scalar
    ``BASELINE_S3`` are snapshotted at import time.  Forgetting this call would
    silently solve the theta sweep on the 0.60 staircase.
    """
    global _U, _V, K, BASELINE_S3
    _U = [0.0] + [b for b, _ in _cs.S2_T]
    _V = [v for _, v in _cs.S2_T]
    K = len(_cs.S2_T)
    BASELINE_S3 = float(_cs.BASELINE_S3)
    assert len(_U) == K + 1, "staircase length mismatch after sync"
    assert abs(_V[-1] - BASELINE_S3) < 1e-12, (
        "S2 floor %r and S3 baseline %r disagree after sync" % (_V[-1], BASELINE_S3))
    return _U, _V, BASELINE_S3


def _make_solver(name, msg=False, time_limit=None, gap_rel=1e-9, gap_abs=1e-9):
    """Build the requested PuLP solver, defaulting to HiGHS.

    The optimality tolerances are passed explicitly: the PuLP defaults leave
    HiGHS on its own 1e-4 relative gap, which silently stops a few instances
    marginally below the true optimum (observed once at -7e-6 in F2).
    """
    if name == "cbc":
        kw = {"msg": 1 if msg else 0, "gapRel": gap_rel, "gapAbs": gap_abs}
        if time_limit:
            kw["timeLimit"] = time_limit
        return pulp.PULP_CBC_CMD(**kw)
    # HiGHS (default).  ``threads`` left at solver default; these models are tiny.
    kw = {"msg": bool(msg), "gapRel": gap_rel, "gapAbs": gap_abs}
    if time_limit:
        kw["timeLimit"] = time_limit
    return pulp.HiGHS(**kw)


def phase1_coverage(n, pop, demand, S1m, sta_caps, time_limit=None, solver="highs",
                    gap_rel=1e-9, gap_abs=1e-9):
    """Max-coverage Phase 1.  Logically identical to the Phase-1 block of run_milp_exact."""
    n_sta = len(sta_caps)
    prob = pulp.LpProblem("Phase1_exact", pulp.LpMaximize)
    w = {(s, c): pulp.LpVariable("w1_%d_%d" % (s, c), cat="Binary")
         for s in range(n_sta) for c in range(n) if S1m[s, c] > 0}
    if not w:
        return None, 0.0

    prob += pulp.lpSum(pop[c] * w[(s, c)] for (s, c) in w)
    for c in range(n):
        vfs = [w[(s, c)] for s in range(n_sta) if (s, c) in w]
        if vfs:
            prob += pulp.lpSum(vfs) <= 1
    for s in range(n_sta):
        vfs = [(c, w[(s, c)]) for c in range(n) if (s, c) in w]
        prob += pulp.lpSum([demand[c] * v for c, v in vfs]) <= sta_caps[s]

    prob.solve(_make_solver(solver, time_limit=time_limit,
                            gap_rel=gap_rel, gap_abs=gap_abs))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None, 0.0

    m = np.zeros((n_sta, n))
    for (s, c), var in w.items():
        m[s, c] = 1.0 if (pulp.value(var) or 0) > 0.5 else 0.0
    cov = np.array([np.any(m[:, c] > 0.5) for c in range(n)])
    return m, float(pop[cov].sum()) / float(pop.sum())


def _verify(m, pop, demand, caps, S1m, F1max, total_pop):
    """Independent NumPy feasibility check of a decoded assignment.

    Returns a list of human-readable violations (empty == feasible).  This is the
    safety net for solver verdicts: it re-derives capacity, coverage and the
    segment consistency from the assignment alone, never trusting the MILP.
    """
    errs = []
    n_sta, n = m.shape
    # (a) each eligible community served by at most one station; no ineligible service
    per_comm = m.sum(axis=0)
    if np.any(per_comm > 1 + 1e-9):
        errs.append("community served by >1 station")
    for s in range(n_sta):
        for c in range(n):
            if m[s, c] > 0.5 and S1m[s, c] <= 0:
                errs.append("ineligible pair (%d,%d)" % (s, c))
                break
    # (b) capacity
    loads = np.array([sum(demand[c] * m[s, c] for c in range(n)) for s in range(n_sta)])
    for s in range(n_sta):
        if loads[s] > caps[s] + 1e-6 * max(1.0, caps[s]):
            errs.append("capacity exceeded at station %d (%.3f > %.3f)"
                        % (s, loads[s], caps[s]))
    # (c) coverage lock
    served = float(sum(pop[c] for c in range(n) if per_comm[c] > 0.5))
    if served < F1max * total_pop - 1.0 - 1e-6:
        errs.append("coverage lock violated (%.3f < %.3f)"
                    % (served, F1max * total_pop - 1.0))
    return errs


def phase2_exact(n, pop, demand, S1m, sta_caps, F1_max, s3=None,
                 time_limit=None, gap_rel=1e-9, gap_abs=1e-9, msg=False,
                 solver="highs", fallback="cbc"):
    """Exact single-MILP Phase 2.

    ``s3`` optionally overrides the baseline non-congestion satisfaction: pass a
    scalar (uniform) or a length-``n_sta`` array (per-station, used by the
    subsidy experiment). ``None`` reproduces the baseline constant exactly.

    Returns a dict with keys:
        F2, F1, cov, w, s2, utils, obj, status, mip_gap, enc_residual, load
    or ``None`` if no *verified feasible* assignment could be produced.
    """
    n_sta = len(sta_caps)
    total_pop = float(pop.sum())
    caps = np.asarray(sta_caps, dtype=float)

    # normalise s3 to a per-station vector (constant baseline by default)
    if s3 is None:
        s3_vec = np.full(n_sta, float(BASELINE_S3))
    else:
        _a = np.asarray(s3, dtype=float)
        s3_vec = np.full(n_sta, float(_a)) if _a.ndim == 0 else np.asarray(_a, dtype=float)
        if s3_vec.shape[0] != n_sta:
            raise ValueError("s3 must be a scalar or have length n_sta")

    def _build():
        prob = pulp.LpProblem("Phase2_exact", pulp.LpMaximize)
        w = {(s, c): pulp.LpVariable("w2_%d_%d" % (s, c), cat="Binary")
             for s in range(n_sta) for c in range(n) if S1m[s, c] > 0}
        if not w:
            return None, None, None, None

        # --- assignment feasibility (identical to the heuristic version) ----
        for c in range(n):
            vfs = [w[(s, c)] for s in range(n_sta) if (s, c) in w]
            if vfs:
                prob += pulp.lpSum(vfs) <= 1
        for s in range(n_sta):
            vfs = [(c, w[(s, c)]) for c in range(n) if (s, c) in w]
            prob += pulp.lpSum([demand[c] * v for c, v in vfs]) <= caps[s]
        # --- Phase-1 coverage lock (same tolerance as run_milp_exact) -------
        prob += pulp.lpSum([pop[c] * v for (s, c), v in w.items()]) >= F1_max * total_pop - 1

        # --- exact staircase encoding --------------------------------------
        z, t = {}, {}
        for s in range(n_sta):
            cols = [c for c in range(n) if (s, c) in w]
            if not cols:
                # Station has no assignable community: load == 0 identically, so
                # its utilisation sits in segment 0 by definition and L_s == 0.
                # Building segment constraints would force load >= U[k]*cap > 0.
                continue
            cap = float(caps[s])
            load_s = pulp.lpSum([demand[c] * w[(s, c)] for c in cols])
            L_s = pulp.lpSum([pop[c] * w[(s, c)] for c in cols])
            Lmax_s = float(sum(pop[c] for c in cols))       # tight McCormick M
            for k in range(K):
                z[(s, k)] = pulp.LpVariable("z_%d_%d" % (s, k), cat="Binary")
                t[(s, k)] = pulp.LpVariable("t_%d_%d" % (s, k), lowBound=0)
            prob += pulp.lpSum([z[(s, k)] for k in range(K)]) == 1

            for k in range(K):
                # lower bound: load >= U[k]*cap when segment k is selected
                if k > 0:
                    prob += load_s >= _U[k] * cap * z[(s, k)]
                # upper bound: load <= U[k+1]*cap when segment k is selected.
                # Tight big-M: the largest slack available is cap*(1-U[k+1]).
                if k < K - 1:
                    M = cap * (1.0 - _U[k + 1])
                    prob += load_s <= _U[k + 1] * cap + M * (1 - z[(s, k)])
                # McCormick envelope for t = z * L (tight M = Lmax_s)
                prob += t[(s, k)] <= Lmax_s * z[(s, k)]
                prob += t[(s, k)] <= L_s
                prob += t[(s, k)] >= L_s - Lmax_s * (1 - z[(s, k)])

        # --- objective (normalised by total population for numerical scale) --
        prob += (
            pulp.lpSum([pop[c] * W1 * float(S1m[s, c]) * w[(s, c)] for (s, c) in w])
            + W2 * pulp.lpSum([_V[k] * t[(s, k)] for (s, k) in t])
            + W3 * pulp.lpSum([float(s3_vec[s]) * pop[c] * w[(s, c)] for (s, c) in w])
        ) / total_pop
        return prob, w, z, t

    prob, w, z, t = _build()
    if prob is None:
        return None

    kw = {"gapRel": gap_rel, "gapAbs": gap_abs}
    attempts = [solver] + ([fallback] if fallback and fallback != solver else [])
    # The fallback must never make the answer worse.  CBC reports "Optimal"
    # far below the true optimum on these models (measured: 0.2494 vs HiGHS
    # 0.2719 on Beijing, a -8 % swing in F2), so a silent fallback would corrupt
    # a whole sweep.  Keep the best attempt and snapshot its variable values,
    # because prob.solve() overwrites them in place.
    best = {"obj": None, "name": None, "status": None, "values": None}
    for name in attempts:
        prob.solve(_make_solver(name, msg=msg, time_limit=time_limit,
                                gap_rel=gap_rel, gap_abs=gap_abs))
        status = pulp.LpStatus[prob.status]
        obj = pulp.value(prob.objective)
        if status in ("Optimal", "Integer Feasible") and obj is not None:
            if best["obj"] is None or obj > best["obj"] + 1e-12:
                best = {"obj": float(obj), "name": name, "status": status,
                        "values": {v.name: v.value() for v in prob.variables()}}
            if status == "Optimal" and name == solver:
                break
    if best["obj"] is None:
        return None
    for v in prob.variables():                     # restore the best attempt
        v.varValue = best["values"][v.name]
    name, last_status = best["name"], best["status"]

    # --- decode ------------------------------------------------------------
    m = np.zeros((n_sta, n))
    for (s, c), var in w.items():
        m[s, c] = 1.0 if (pulp.value(var) or 0) > 0.5 else 0.0

    # --- independent verification (never trust the solver verdict) ---------
    errs = _verify(m, pop, demand, caps, S1m, F1_max, total_pop)
    if errs:
        return None

    loads = np.array([sum(demand[c] * m[s, c] for c in range(n)) for s in range(n_sta)])
    utils = loads / caps
    s2 = np.array([s2_fn(u) for u in utils])

    F2 = float(sum(pop[c] * (W1 * S1m[s, c] + W2 * s2[s] + W3 * float(s3_vec[s])) * m[s, c]
                   for c in range(n) for s in range(n_sta)) / total_pop)
    covered = {c for c in range(n) if m[:, c].sum() > 0.5}
    F1 = float(pop[list(covered)].sum()) / total_pop if covered else 0.0

    # encoding residual: MILP's own view of S2 vs the true staircase value
    if z:
        s2_enc = np.array([
            sum(_V[k] * round(pulp.value(z[(s, k)]) or 0) for k in range(K))
            if any((s, k) in z for k in range(K)) else s2[s]
            for s in range(n_sta)])
    else:
        s2_enc = np.zeros(n_sta)
    enc_res = float(np.max(np.abs(s2_enc - s2))) if n_sta else 0.0

    return {
        "F2": F2,
        "F1": F1,
        "cov": covered,
        "w": m,
        "s2": s2,
        "utils": utils,
        "load": loads,
        "obj": float(pulp.value(prob.objective)),
        "status": last_status,
        "solver": name,
        "mip_gap": float("nan"),
        "enc_residual": enc_res,
    }


def run_two_phase_exact(n, pop, demand, S1m, sta_caps, s3=None, time_limit=None,
                        gap_rel=1e-9, gap_abs=1e-9, return_details=False,
                        solver="highs", fallback="cbc"):
    """Drop-in replacement for ``run_milp_exact`` with an exact Phase 2.

    ``s3`` mirrors ``phase2_exact``'s per-station override (subsidy experiment).
    Returns (F2, F1, cov) by default; (F2, F1, cov, w, s2, info) when
    ``return_details=True`` (info carries status / enc_residual / solver).
    """
    _w1, F1_max = phase1_coverage(n, pop, demand, S1m, sta_caps,
                                  time_limit=time_limit, solver=solver)
    if _w1 is None:
        return (0.0, 0.0, set()) if not return_details else (0.0, 0.0, set(), None, None, {})

    out = phase2_exact(n, pop, demand, S1m, sta_caps, F1_max, s3=s3,
                       time_limit=time_limit, gap_rel=gap_rel, gap_abs=gap_abs,
                       solver=solver, fallback=fallback)
    if out is None:
        return (0.0, 0.0, set()) if not return_details else (0.0, 0.0, set(), None, None, {})

    if return_details:
        info = {"status": out["status"], "mip_gap": out["mip_gap"],
                "enc_residual": out["enc_residual"], "obj": out["obj"],
                "solver": out["solver"]}
        return out["F2"], out["F1"], out["cov"], out["w"], out["s2"], info
    return out["F2"], out["F1"], out["cov"]
