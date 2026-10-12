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

## 3. Experiment 1 — 21 real cities (§5) — 15–30 min

```
python code/03_experiments/exp1_21cities.py
python code/03_experiments/exp1_regression.py      # Table 4, §5.4
python code/03_experiments/exp1_robustness.py      # §5.5
```

**Expected values (locked, two-phase specification, n = 21):**

| Quantity | Value |
|---|---|
| mean PoA | ≥ 1.0267 |
| max PoA (Chongqing) | ≥ 1.0663 |
| violations of the universal bound 1.667 | 0 |
| regression: connectivity β (locked) | +1.3396 (p = 0.384) |
| regression: log(n_sta) β (locked) | +0.0183 (p = 0.031) |
| regression: R² (locked) | 0.516 |

Outputs → `results/main/`.

---

## 4. Experiment 2 — full enumeration (§6) — 60–120 min with 8 workers

```
python code/03_experiments/exp2_enumeration.py --workers 8
python code/03_experiments/exp2_channel_decomp.py    # §6.3, Table 8
python code/03_experiments/exp2_layered_bounds.py    # §6.4, Table 1
python code/03_experiments/exp2_connectivity_reg.py  # §6.6
```

### 4.1 The design reproduces exactly

```
python code/02_core_solver/instance_generator.py
```

prints

```
valid configurations: 15207
```

The design is: 10 candidate sites × 4 size tiers (construction cost
18 / 32 / 45) with total construction cost ≤ 118, minus the single all-zero
assignment (no station built ⇒ zero capacity ⇒ PoA undefined). 15,208 − 1 =
**15,207**, the number reported in the paper.

### 4.2 Calibration of the synthetic geometry

The synthetic enumeration has exactly one free geometric parameter: the side
length of the synthetic city, `city_side_m` in
`code/02_core_solver/instance_generator.py`. It controls how much the service
areas of neighbouring stations overlap, hence how much strategic congestion the
decentralized equilibrium generates.

Measured on a random sample of 18 configurations (seed 2024):

| `city_side_m` | mean PoA | median PoA | max PoA | share PoA > 1 |
|---|---|---|---|---|
| 1600 | 1.0845 | 1.0528 | 1.2617 | 0.94 |
| 2200 *(shipped default)* | 1.0697 | 1.0375 | 1.3031 | 0.78 |
| 2400 | 1.0320 | 1.0000 | 1.1366 | 0.44 |
| 2800 | 1.0267 | 1.0000 | 1.1366 | 0.39 |
| **published (§6)** | **1.0398** | **1.0336** | **1.4398** | **0.878** |

The shipped default (2200 m) reproduces the published **median** (1.0375 vs
1.0336) and the **share of configurations with PoA > 1** (0.78 vs 0.878) to
within sampling error. Reproduce the table with:

```
python code/03_experiments/exp2_enumeration.py --side 2200 --limit 200 --workers 4
```

### 4.3 What is and is not bit-reproducible

| Result | Status |
|---|---|
| The 15,207-configuration design space | **exactly reproducible** |
| Sign and ordering of the three channels (coverage > assignment > congestion < 0) | **reproducible** |
| The qualitative findings of §6.3–§6.4 | **reproducible** |
| The four-decimal statistics of Table 5 (mean 1.0398, median 1.0336, max 1.4398) | **approximately** — see §4.2 |

The archived per-configuration output of the original run is derived from the
restricted competition specification and is therefore not redistributed (see
`data/external/DATA_SOURCES.md`, §2.2). Consequently the last row cannot be
verified to four decimals from this package alone; the qualitative claims and
every equation can.

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

## 6. Sensitivity analyses — 30–45 min

```
python code/04_sensitivity/capacity_heterogeneity.py   # §6.5
python code/04_sensitivity/budget_sensitivity.py       # §6.5
python code/04_sensitivity/cluster_radius.py           # §5.5
python code/04_sensitivity/gamma_sensitivity.py        # appendix
python code/04_sensitivity/s2_robustness.py            # appendix
```

**Expected:** PoA is insensitive to the budget (range 0.0108 over
25 / 30 / 36 / 40 万元 per station) and increasing in capacity heterogeneity
(gini 0.0424 → 0.1955 → 0.2487 gives PoA 1.0291 → 1.0394 → 1.0438).

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
