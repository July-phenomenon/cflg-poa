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
| `code/04_sensitivity/` | Capacity heterogeneity, budget, cluster radius, γ and S₂ robustness (§5.5, §6.5, appendices) |
| `code/05_figures_tables/` | One entry point per figure and per table |
| `code/utils/` | Shared constants, plotting template, I/O helpers |
| `data/processed/` | 21-city parameters (derived from publicly available rasters) |
| `results/figures/` | The four published figures + graphical abstract (reference copies) |
| `results/tables/` | The published table bodies as CSV (reference copies) |
| `docs/` | Reproduction guide, environment notes, troubleshooting, checklist |
| `tests/` | Unit tests for the solver and the satisfaction functions |

**Solver:** [PuLP](https://github.com/coin-or/PuLP) with the bundled
**CBC** mixed-integer solver.  CBC is open source; no licence key, no
registration, no network access is required.  No commercial solver is used
anywhere in this project, so the package is fully reproducible with free
tooling.

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

Every script can also be run individually from the repository root, e.g.

```bash
python code/03_experiments/exp1_21cities.py
python code/05_figures_tables/fig1_poa21.py
```

Scripts locate the repository root through the `CFLG_ROOT` environment
variable and fall back to the directory two levels above the script, so no
absolute path is hardcoded anywhere.

---

## 3. Time budget

Measured on an 8-core desktop (single process unless stated).

| Step | Command | Expected time |
|---|---|---|
| Environment | `pip install -r requirements.txt` | 5–10 min |
| Parameter derivation | `code/01_parameter_derivation/` | 2–5 min |
| Experiment 1 (21 cities) | `exp1_21cities.py` | 15–30 min |
| Experiment 2 (15,207 configs) | `exp2_enumeration.py --workers 8` | 60–120 min |
| Experiment 3 (subsidy) | `exp3_subsidy.py` | 10–20 min |
| Sensitivity analyses | `code/04_sensitivity/` | 30–45 min |
| Figures and tables | `code/05_figures_tables/` | 2–5 min |
| **Total** | | **≈ 2–3.5 h** |

A smoke test that exercises the whole pipeline without the long sweep:

```bash
python code/03_experiments/exp2_enumeration.py --limit 200 --workers 4
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
| Competition specification (station sizes, capacities, cost table, satisfaction ladders) | **not included** – available from the corresponding author on reasonable request | restricted |
| Full enumeration output (15,207 configurations) | **not included** – regenerate with `code/03_experiments/exp2_enumeration.py` | restricted |

The parameters that come from the competition specification are nonetheless
recorded verbatim in `code/utils/constants.py` (weights `W1/W2/W3`,
satisfaction ladders, capacity and cost tiers), so a reader can check every
equation without access to the restricted documents.

Full details: [`data/external/DATA_SOURCES.md`](data/external/DATA_SOURCES.md).

---

## 6. Reproducing specific paper results

| Paper item | Command | Output |
|---|---|---|
| Table 1 (layered bounds) | `python code/05_figures_tables/table1_layered.py` | `results/tables/Table6_2_layered_stats.csv` |
| Table 2 (21-city parameters) | `python code/05_figures_tables/table2_city_params.py` | `results/tables/Table5_1_21cities_params.csv` |
| Table 3 (21-city PoA) | `python code/05_figures_tables/table3_poa21.py` | `results/tables/Table5_2_21cities_PoA.csv` |
| Table 4 (regression) | `python code/05_figures_tables/table4_regression.py` | regression output, §5.4 |
| Table 5 (enumeration distribution) | `python code/05_figures_tables/table5_poa_dist.py` | `results/tables/Table6_1_enum_stats.csv` |
| Table 6 (subsidy comparison) | `python code/05_figures_tables/table6_subsidy.py` | `results/tables/Table7_1_subsidy_comparison.csv` |
| Table 7 (three specifications) | `python code/05_figures_tables/table7_three_specs.py` | `results/tables/TableA_1_three_specs.csv` |
| Table 8 (channel contributions) | `python code/05_figures_tables/table8_channels.py` | channel decomposition |
| Table 9 (full subsidy table) | `python code/05_figures_tables/table9_full_subsidy.py` | appendix B |
| Figure 1 | `python code/05_figures_tables/fig1_poa21.py` | `results/figures/figure1.pdf` |
| Figure 2 | `python code/05_figures_tables/fig2_poa_dist.py` | `results/figures/figure2.pdf` |
| Figure 3 | `python code/05_figures_tables/fig3_channel.py` | `results/figures/figure3.pdf` |
| Figure 4 | `python code/05_figures_tables/fig4_subsidy.py` | `results/figures/figure4.pdf` |
| Graphical abstract | `python code/05_figures_tables/fig_ga.py` | `results/figures/GraphicalAbstract.pdf` |

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
* **No absolute paths.** All I/O goes through `CFLG_ROOT` / package-relative
  defaults.
* **No hidden state.** Intermediate results are plain JSON/CSV.

Known limits (documented in full in `docs/REPRODUCIBILITY.md`):

* The enumeration *design* (15,207 configurations) is reproduced exactly.
  The synthetic instance geometry has one calibrated knob (the city side).
* Because the raw competition inputs and the archived enumeration output are
  not redistributed, the four-decimal statistics of §6 are reproduced to the
  accuracy stated in `docs/REPRODUCIBILITY.md`, not bit-for-bit.

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
```
