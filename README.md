# Bounded Price of Anarchy in Capacitated Facility Location Games with Congestion Feedback

Reproduction package for the manuscript *"Bounded Price of Anarchy in
Capacitated Facility Location Games with Congestion Feedback"* (Bei Cheng).

**Repository:** <https://github.com/July-phenomenon/cflg-poa>

The package contains everything needed to re-derive every number, table and
figure in the paper **except** the raw competition inputs and the artefacts
directly derived from them, which are not redistributed (see
[Data sources](#data-sources)).

---

## 1. What is in the box

| Directory | Contents |
|---|---|
| `code/01_parameter_derivation/` | Derivation of the 21-city instance parameters from the accessibility rasters (§5.1) |
| `code/02_core_solver/` | Two-phase MILP (§3.3), Nash best-response (§3.4), satisfaction functions (§3.2), synthetic instance generator (§3.1, §6) |
| `code/03_experiments/` | Experiment 1 (21 cities, §5), experiment 2 (full enumeration, §6), experiment 3 (subsidy, §7) |
| `code/04_sensitivity/` | Capacity heterogeneity, budget, cluster radius (`cluster_radius_21.py`), γ and S₂ robustness (§5.5, §6.5, appendices) |
| `code/05_figures_tables/` | One entry point per figure and per table |
| `code/utils/` | Shared constants, plotting template, I/O helpers |
| `data/processed/` | 21-city parameters (derived from publicly available rasters) |
| `results/figures/` | The four published figures + graphical abstract (reference copies) |
| `results/tables/` | The published table bodies as CSV (reference copies) |
| `results/main/` | Exact single-MILP benchmark JSONs — **regenerated locally by the experiment scripts**, not shipped (see §5). `results/enumeration_poa_results.csv` (one row per configuration) *is* shipped |
| `docs/` | Reproduction guide, environment notes, troubleshooting, checklist |
| `tests/` | Unit tests for the solver and the satisfaction functions |

**Solver:** [PuLP](https://github.com/coin-or/PuLP) with the **HiGHS** MILP solver
bundled with PuLP 3.x.  HiGHS is used for the exact single-MILP Phase 2 of every
result reported in the paper; CBC (also bundled with PuLP) is kept only as an
optional cross-check (`solver="cbc"`), because CBC 2.10 rejects some of these
instances as *infeasible* while HiGHS and CBC-on-LP both return the correct
optimum (see the header of `code/02_core_solver/core_solver_exact.py`).  Both
solvers are open source; no licence key, registration or network access is
required.

---

## 2. Quick start

```bash
conda env create -f environment.yml && conda activate cflg-poa
# or
pip install -r requirements.txt

bash scripts/run_all.sh          # Linux / macOS / Git-Bash
# or
powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1
```

Every script can also be run individually from the repository root.  The exact
(the paper's) chain is:

```bash
python code/03_experiments/exp2_enumeration_exact.py                       # 15,207 configs
python code/03_experiments/exp1_21cities.py --exact-phase2 --nash-mode locked \
       --budget-per-station 36 --output results/main/21cities_exact_locked.json
python code/03_experiments/exp1_21cities.py --exact-phase2 --nash-mode classic \
       --budget-per-station 36 --output results/main/21cities_exact_classic.json
python code/03_experiments/exp3_subsidy.py --exact-phase2                  # subsidy
python code/03_experiments/exp1_regression.py                              # Table 4 + LOO
python code/04_sensitivity/cap_floor_sensitivity.py --exact-phase2 ...     # theta sweep
python code/05_figures_tables/export_exact_figure_inputs.py                # figure inputs
python code/05_figures_tables/make_figures_paper.py                        # the 4 figures + GA
python tests/verify_paper_numbers.py                                       # 102-number audit
```

`tests/verify_paper_numbers.py` recomputes every number quoted in the manuscript
from the exact artefacts and exits non-zero on any mismatch, so it doubles as a
regression test for the paper itself.

Scripts locate the repository root through the `CFLG_ROOT` environment
variable and fall back to the directory two levels above the script, so no
absolute path is hardcoded anywhere.

---

## 3. Time budget

Measured on an 8-core desktop (single process unless stated; exact solver, 2026-10).

| Step | Command | Expected time |
|---|---|---|
| Environment | `pip install -r requirements.txt` | 5–10 min |
| Parameter derivation | `code/01_parameter_derivation/` | 2–5 min |
| Experiment 1 (21 cities, exact) | `exp1_21cities.py --exact-phase2 --phase2-time-limit 180` | 1.5–2.5 h |
| Experiment 2 (15,207 configs, exact) | `exp2_enumeration_exact.py` | 2–3 min |
| Experiment 3 (subsidy, exact) | `exp3_subsidy.py --exact-phase2` | 8–15 min |
| Regression + leave-one-out | `exp1_regression.py` | < 1 min |
| Sensitivity: theta sweep | `cap_floor_sensitivity.py --exact-phase2` | 3 min per grid |
| Cluster-radius cells (optional) | `cluster_radius_21.py` | 1.5–2.5 h per factor |
| Figures and tables | `export_exact_figure_inputs.py` + `make_figures_paper.py` | < 1 min |
| Paper audit | `tests/verify_paper_numbers.py` | < 1 min |
| **Total** | | **≈ 2–3 h** |

The 21-city cells are the only slow step and they dominate the budget: with no
`--phase2-time-limit` a single city can take well over 20 minutes of wall time,
so the 180 s cap is what makes a full sweep tractable. All 21 cities still reach
solver optimality with that cap. Every city is checkpointed, so an interrupted
sweep resumes rather than restarts.

A smoke test that exercises the whole pipeline without the long sweeps:

```bash
python code/03_experiments/exp2_enumeration_exact.py --limit 200
python code/03_experiments/exp1_21cities.py --exact-phase2 --limit 2
```

---

## 4. Ordering

The pipeline is a DAG; run it in this order (this is exactly what
`scripts/run_all.sh` does):

```
01 parameter derivation
        |
        v
exp1 (21 cities)  ---->  tables 2, 3, 4 ; figures 1
        |
        v
exp2 (enumeration) ----> tables 1, 5, 8 ; figures 2, 3 ; graphical abstract
        |
        v
exp3 (subsidy)  ------>  tables 6, 9 ; figure 4
        |
        v
04 sensitivity  ------>  appendix numbers
```

---

## 5. Data sources

| Data | Availability | Licence |
|---|---|---|
| 21-city accessibility rasters | Han et al. (2026), Zenodo DOI `10.5281/zenodo.17010522` (public) | CC BY 4.0 |
| Derived 21-city parameters | **included**, `data/processed/21city_parameters.json` | CC BY 4.0 |
| Competition specification (station sizes, capacities, cost table, satisfaction ladders; CSEE Cup National Undergraduate Mathematical Modeling Competition, 18th edition, 2026) | **not included** – its redistribution is restricted by the competition terms | restricted |
| Raw enumeration output (full per-configuration JSON, 15,208 records incl. the degenerate all-zero one) | **not included** – regenerate with `code/03_experiments/exp2_enumeration.py` | restricted |
| **Processed** enumeration results (PoA per configuration) | **included**, `results/enumeration_poa_results.csv` (15,207 rows: `config_id, PoA, F2_MILP, F2_Nash`) | CC BY 4.0 |

Raw competition inputs (the problem statement and its attachments) and the
artefacts directly derived from them are **not** redistributed here, because
their redistribution is restricted by the competition terms; only the scripts
that regenerate them are provided. Processed results that do not expose any
competition input (configuration-level PoA values and aggregate statistics) are
included in this package, and any further processed results are available from
the corresponding author on reasonable request.

The processed CSV contains only the two objectives and their ratio, so it can
be redistributed freely; it is produced from the raw enumeration output by

```bash
python code/03_experiments/export_derived_results.py --src <enumeration JSON>
```

No competition input parameter appears in it.

The parameters that come from the competition specification are nonetheless
recorded verbatim in `code/utils/constants.py` (weights `W1/W2/W3`,
satisfaction ladders, capacity and cost tiers), so a reader can check every
equation without access to the restricted documents.

Full details: [`data/external/DATA_SOURCES.md`](data/external/DATA_SOURCES.md).

---

## 6. Reproducing specific paper results

| Paper item | Source artefact | Regenerate with |
|---|---|---|
| Table 1 (layered bounds) | `results/main/results_enumeration_exact.json` | `python code/03_experiments/exp2_layered_bounds.py` |
| Table 2 (21-city parameters) | `data/processed/21city_parameters.json` | `python code/05_figures_tables/table2_city_params.py` |
| Table 3 (21-city PoA) | `21cities_exact_locked.json` | `python code/05_figures_tables/export_exact_figure_inputs.py` |
| Table 4 (regression + LOO) | `21cities_exact_locked.json` | `python code/03_experiments/exp1_regression.py` |
| Table 5 (enumeration distribution) | `results_enumeration_exact.json` | `python code/05_figures_tables/table5_poa_dist.py` |
| Table 6 (subsidy comparison) | `_subsidy_exact/subsidy_exact.json` | `python code/05_figures_tables/export_exact_figure_inputs.py` |
| Table 7 (three specifications) | the two 21-city JSONs | `python code/05_figures_tables/table7_three_specs.py` |
| Table 8 (channel contributions) | `results_enumeration_exact.json` | `python code/03_experiments/exp2_enumeration_channels.py` |
| Table 9 (full subsidy table) | `_subsidy_exact/subsidy_exact.json` | `python code/05_figures_tables/table9_full_subsidy.py` |
| Figures 1–4, graphical abstract | exact JSONs + `results/tables/figure_inputs/*.csv` | `python code/05_figures_tables/make_figures.py` then `make_figures_paper.py` |
| Every number in the text | all of the above | `python tests/verify_paper_numbers.py` |

Step-by-step instructions, expected numerical values and the list of results
that can and cannot be bit-reproduced are in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

---

## 7. Reproducibility guarantees

* **Deterministic.** Geometry seeds are derived with `hashlib.md5` on the city
  name (`seed_for` in `code/02_core_solver/core_solver.py`) rather than
  `hash()`, so they do not depend on `PYTHONHASHSEED`. The enumeration uses a
  fixed seed (`2024`). No experiment depends on wall-clock time or on the
  order of dictionary iteration.
* **Optimality-certified.** Every Phase-2 benchmark is a single HiGHS MILP with
  a 1e-9 relative gap, and each solution is re-verified in plain NumPy
  (capacity, coverage lock, staircase consistency) before it enters a result
  file. No PoA is a lower bound.
* **Audited.** `tests/verify_paper_numbers.py` recomputes 102 published numbers
  from the artefacts and fails loudly on any drift.
* **No absolute paths.** All I/O goes through `CFLG_ROOT` / package-relative
  defaults.
* **No hidden state.** Intermediate results are plain JSON/CSV, written
  atomically; every long run also keeps a per-city checkpoint.

Known limits (documented in full in `docs/REPRODUCIBILITY.md`):

* The enumeration *design* (15,207 configurations) is reproduced exactly.
  The synthetic instance geometry has one calibrated knob (the city side).
* Because the raw competition inputs and the archived enumeration output are
  not redistributed, the four-decimal statistics of §6 are reproduced to the
  accuracy stated in `docs/REPRODUCIBILITY.md`, not bit-for-bit.
* Two independent exact runs of the enumeration (same geometry, different MILP
  gap tolerance) agree on 15,206 of 15,207 configurations; the single
  disagreement is a near-tie in which `F2_MILP` moves by 1.5e-5, i.e. far below
  the four-decimal reporting precision. Every aggregate statistic in the paper
  is unchanged at four decimals.

---

## 8. Licence and citation

Code: MIT. Processed data: CC BY 4.0. See [`LICENSE`](LICENSE).

```bibtex
@article{cheng2026bounded,
  title   = {Bounded Price of Anarchy in Capacitated Facility Location Games
             with Congestion Feedback},
  author  = {Cheng, Bei},
  journal = {To be added at submission},
  year    = {2026}
}

---

## 9. Extensions added after the core pipeline

The core pipeline above reproduces the original nine tables and four figures.
The following artefacts were added later (E1–E10 experiments, the E8 external
validation, and the 21-city taxonomy) and are regenerated by their own scripts:

| Paper item | Artefact | Regenerate with |
|---|---|---|
| Table 4 (coverage bands, §5.6) | `results/tables/T13_coverage_bands.{json,csv,tex}` | `python code/05_figures_tables/table_cov_bands.py` |
| Fig. 2 (coverage–PoA + heterogeneity, §5.6 / E3, E10) | `results/figures/fig5_coverage_poa.{pdf,png}` | `python code/05_figures_tables/fig5_coverage_poa.py` |
| Appendix C (sensitivity audit, E7) | `results/tables/T14_sensitivity_audit.{json,csv}` | `python code/04_sensitivity/audit_sensitivity_provenance.py` |
| Appendix C (regression ladder, E10) | `results/tables/T15_enum_regression_ladder.{json,md}` | `python code/03_experiments/exp2_enum_regression_full.py` |
| Fig. 6 / Table (21-city data-driven taxonomy, §5.6) | `results/figures/fig6_city_taxonomy.{pdf,png}`, `results/tables/T16_city_taxonomy.{json,csv}` | `python code/05_figures_tables/classify_21cities.py` |
| §5.6 external validation (E8, Wang2026 1-km data) | `results/e8_wang2026/*` | `python code/06_e8_wang2026/closed_loop_check.py` (after `run_all_21.py`) |

`code/06_e8_wang2026/` holds the three-step parameter derivation specialised to
the newer 1-km source, the 21-city external-validation run, and the closed-loop
correlation check. All extensions write only under `results/` and never modify
the core pipeline's inputs, so the original nine-table / four-figure reproduction
is unaffected. A consolidated index of every extension script, its artefact, and
the manuscript paragraph it supports is in `README_E1E3E7E10.md`.
```
