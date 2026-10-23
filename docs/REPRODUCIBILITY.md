# Reproduction guide

Everything below is run from the repository root. Times are for an 8-core
desktop; the enumeration step dominates.

```
export CFLG_ROOT=$PWD        # optional; scripts also infer it
```

---

## 0. Determinism

| Concern | Mechanism |
|---|---|
| Geometry seeds | `seed_for(name)` in `code/02_core_solver/core_solver.py` uses `hashlib.md5(name)`, **not** `hash(name)`. Python salts `hash()` per process, which previously made each re-run draw different geometry for the same city. |
| Enumeration geometry | fixed `SEED_GEOMETRY = 2024` |
| Sampling / shuffling | no experiment uses unseeded randomness |
| MILP backend | CBC (bundled with PuLP). CBC is deterministic for a fixed model and a fixed binary. |
| Nash iteration | fixed iteration cap and convergence tolerance, deterministic given the instance |

---

## 1. Environment (5–10 min)

```
conda env create -f environment.yml && conda activate cflg-poa
```

or `pip install -r requirements.txt`. See `docs/ENVIRONMENT.md`.

---

## 2. Parameter derivation (§5.1) — 2–5 min, optional

```
python code/01_parameter_derivation/derive_21city_params.py
```

Requires the Han et al. (2026) rasters in `data/raw/han_et_al_2026_rasters/`.
**Skip this if you only want to reproduce the results**: the derived parameters
are shipped in `data/processed/21city_parameters.json`.

---

## 3. Experiment 1 — 21 real cities (§5) — 1.5–2.5 h

```
python code/03_experiments/exp1_21cities.py --exact-phase2 --nash-mode locked \
       --budget-per-station 36 --phase2-time-limit 180 \
       --output results/main/21cities_exact_locked.json
python code/03_experiments/exp1_21cities.py --exact-phase2 --nash-mode classic \
       --budget-per-station 36 --phase2-time-limit 180 \
       --output results/main/21cities_exact_classic.json
python code/03_experiments/exp1_regression.py      # Table 4, LOO, §5.4
python code/03_experiments/exp1_robustness.py      # §5.5
```

**Expected values (coverage-locked, exact Phase 2, n = 21):**

| Quantity | Value |
|---|---|
| mean PoA | 1.0376 |
| median PoA | 1.0357 |
| max PoA (Chongqing) | 1.0791 |
| min PoA (Dongguan) | 1.0000 |
| MILP strictly better / tie / Nash better | 18 / 3 / 0 |
| violations of the universal bound 1.667 | 0 |
| classic-specification mean / max | 1.0503 / 1.1604 |
| regression: connectivity β (locked) | +2.1523 (p = 0.123) |
| regression: log(n_sta) β (locked) | +0.0246 (p = 0.0023) |
| regression: R² (locked) | 0.7025 |
| leave-one-city-out: log(n_sta) significant | 21/21 (coef +0.0175 … +0.0363) |

Each city is checkpointed after it finishes (`<output>.ckpt.json`), so an
interrupted run resumes where it stopped; the checkpoint records the full
specification (budget, Nash mode, sizing rule, time limit, cluster factor) and
is discarded automatically if any of those change.

Outputs → `results/main/`.

---

## 4. Experiment 2 — full enumeration (§6) — 2–3 min

```
python code/03_experiments/exp2_enumeration_exact.py          # exact Phase 2
python code/03_experiments/exp2_enumeration_channels.py       # channel decomposition
python code/03_experiments/exp2_layered_bounds.py             # layered bounds
python code/03_experiments/exp2_enum_connectivity_reg.py      # connectivity regression
```

### 4.1 The design reproduces exactly

```
python code/02_core_solver/instance_generator.py
```

prints

```
valid configurations: 15207
```

The design is: 10 candidate sites x 4 size tiers (construction cost
18 / 32 / 45) with total construction cost <= 118, minus the single all-zero
assignment (no station built => zero capacity => PoA undefined). 15,208 - 1 =
**15,207**, the number reported in the paper.

### 4.2 The statistics reproduce exactly

Since the 2026-10 revision the primary enumeration and the sensitivity sweeps
share one canonical geometry (`base_geometry(seed=2024, side=2200 m)`), and
Phase 2 is a single exact MILP instead of an outer fixed-point loop. The
published statistics are therefore no longer approximations:

| Statistic | Published | Regenerated |
|---|---|---|
| mean / median PoA | 1.0610 / 1.0372 | 1.0610 / 1.0372 |
| p99 / max PoA | 1.3071 / 1.6404 | 1.3071 / 1.6404 |
| MILP strictly better / tied / Nash better | 73.98 % / 17.84 % / 8.18 % | same |
| coverage-locked layer | n = 4,317, mean 1.0092, max 1.1560 | same |
| coverage-unlocked layer | n = 10,890, mean 1.0816, max 1.6404 | same |
| channel median shares | assignment +69.3 %, coverage +43.1 %, congestion -12.6 % | same |

### 4.3 Determinism of the exact MILP

Two independent runs of the full sweep (identical geometry, different MILP gap
tolerance) agree on 15,206 of the 15,207 configurations. The single
disagreement is the near-tie configuration `[2,0,0,0,1,0,0,0,1,2]`, in which
`F2_MILP` moves by 7.0e-6 (and `PoA` by 1.5e-5) — about seven orders of
magnitude above the 1e-12 tie-resolution floor and far below the four-decimal
reporting precision (0.47378 vs 0.47377 both round to 0.4738); every aggregate
statistic above is unchanged, and `F2_Nash` is bit-identical in all 15,207 rows.

A 300-configuration smoke test reproduces the shipped file bit for bit:

```
python code/03_experiments/exp2_enumeration_exact.py --limit 300 --out /tmp/enum_300.json
```

### 4.4 Solver determinism of the 21-city cells

The enumeration of §6 is bit-reproducible (see §4.3).  The 21-city cells are
**not**, and the difference matters at the fourth decimal:

* the instance geometry, the coverage-locked Nash equilibrium and the Phase-1
  coverage are bit-reproducible (verified: identical to the last digit);
* the Phase-2 MILP is solved by HiGHS at a 1e-9 gap under a 180 s per-city cap,
  and HiGHS runs multi-threaded by default.  Five consecutive solves of the
  *same* Beijing problem object returned two different objectives
  (0.27198081 and 0.27199053, spread 9.7e-6), both reported as `Optimal`.
  Across all runs the value has taken three distinct values spanning 4.8e-5.

Consequences: a single city's PoA wobbles by about ±2e-4 between runs
(Beijing: 1.0644-1.0646); the 21-city mean is stable to about ±1e-4; every
theoretical bound and every aggregate statistic is unaffected.

The thread count cannot be pinned in this environment: `pulp.HiGHS(threads=N)`
makes the solver fail outright (even a two-variable toy problem returns
`Not Solved`), and `solverParams={"threads": N}` raises
`setOptionValue(): incompatible function arguments` under the shipped
PuLP 3.3.1 / highspy combination.

`phase2_exact` therefore guards its fallback solver: each attempt's objective
is recorded and only a strictly better one replaces the incumbent, and the
primary solver stops the loop as soon as it reports `Optimal`.  This matters
because CBC alone reaches only 0.2494 on Beijing while still reporting
`Optimal` -- a silent fallback would have corrupted a whole sweep.

### 4.5 What is still not shipped

The raw per-configuration output is not redistributed (it is derived from the
restricted competition specification, see `data/external/DATA_SOURCES.md`,
SS2.2); the scripts regenerate it, and `results/enumeration_poa_results.csv`
gives the processed PoA per configuration.

---

## 5. Experiment 3 — subsidy (§7) — 10–20 min

```
python code/03_experiments/exp3_subsidy.py
python code/03_experiments/exp3_policy_analysis.py   # §7.4, Table 9
```

**Expected (consistent specification):** the V2 subsidy raises decentralized
welfare by 5.97 % on average while leaving the relative gap essentially
unchanged (mean PoA ≥ 1.0394 → ≥ 1.0497); no city's gap is closed.

---

## 6. Sensitivity analyses — 30–45 min (cluster radius: +1.5–2.5 h per factor)

```
python code/04_sensitivity/capacity_heterogeneity.py   # §6.5
python code/04_sensitivity/budget_sensitivity.py       # §6.5
python code/04_sensitivity/cluster_radius_21.py        # §5.5, 21 cities
python code/04_sensitivity/gamma_sensitivity.py        # appendix
python code/04_sensitivity/s2_robustness.py            # appendix
```

`cluster_radius_21.py` is a driver around `exp1_21cities.py --cluster-factor`;
it re-runs all 21 cities with the exact Phase 2 at factors 0.8 / 1.0 / 1.2 and
prints the summary. (`cluster_radius.py` is an unrelated legacy scale probe with
the pre-2026 120 万元 budget; it produces no paper number.)

**Expected (21 cities, coverage-locked, exact Phase 2):**

| cluster factor | mean PoA | median | max | violations of 1.667 |
|---|---|---|---|---|
| 0.8 | 1.0688 | 1.0753 | 1.1764 (Dalian) | 0 |
| 1.0 | 1.0376 | 1.0357 | 1.0791 (Chongqing) | 0 |
| 1.2 | 1.0267 | 1.0216 | 1.0897 | 0 |

The mean moves monotonically with the factor over a band of 0.0421, and every
city reaches solver optimality in every cell.

**Budget / capacity heterogeneity** (approximate Phase 2, archived runs): PoA is
insensitive to the budget (range 0.0108 over 25 / 30 / 36 / 40 万元 per station)
and increasing in capacity heterogeneity (gini 0.0424 → 0.1955 → 0.2487 gives
PoA 1.0291 → 1.0394 → 1.0438).

---

## 7. Figures and tables — 2–5 min

```
python code/05_figures_tables/make_figures.py     # all four figures + GA
python code/05_figures_tables/make_tables.py      # all tables
```

or one at a time:

```
python code/05_figures_tables/fig1_poa21.py
python code/05_figures_tables/table3_poa21.py
...
```

**Figure 1 and Figure 4 need only `data/processed/` and are always
reproducible.** Figure 2, Figure 3 and the graphical abstract need the
enumeration output; if it is absent they print

```
skip fig2_poa_dist: enumeration result file not available
```

and exit cleanly rather than crashing. Point `CFLG_ENUM_JSON` at an existing
result file to override the lookup.

The table scripts perform a pre-flight check and, if `results/main/` is empty,
exit with the exact list of experiment scripts to run first.

---

## 8. Verification

```
python tests/test_solver.py
python tests/test_satisfaction.py
python tests/verify_paper_numbers.py      # requires the archived result files
```

`verify_paper_numbers.py` cross-checks the published constants against the
archived result files listed in `data/external/DATA_SOURCES.md` §2.2. Those
files are not redistributed, so this test is expected to skip with a clear
message in a fresh checkout.
