# Data sources, licences and availability

Every input used by this package is listed below with its provenance, licence
and the exact way it enters the pipeline.

---

## 1. Third-party (public) data

### 1.1 21-city accessibility rasters — Han et al. (2026)

* **What it is:** gridded elderly-care accessibility surfaces for 21 Chinese
  cities, used to derive the community layout, community populations and
  candidate-station coordinates of the 21 real instances (§5.1).
* **Source:** Han et al. (2026), Zenodo, DOI **`10.5281/zenodo.17010522`**.
* **Licence:** CC BY 4.0.
* **Redistribution:** the raw rasters (> 1 GB) are **not** included. They are
  publicly downloadable from the DOI above. Only the *derived parameters* are
  shipped here (`data/processed/21city_parameters.json`), which keeps the
  package small and stays within the licence.

### 1.2 Derived 21-city parameters (shipped)

| File | Contents | Licence |
|---|---|---|
| `data/processed/21city_parameters.json` | per-city `n_com`, `r_c`, `n_sta`, population, area | CC BY 4.0 |
| `data/processed/21city_parameters.csv` | same, tabular | CC BY 4.0 |
| `data/processed/21city_summary_statistics.csv` | cross-city summary | CC BY 4.0 |

These are the only data files needed to reproduce experiment 1 (§5), Figures 1
and 4, and Tables 2–4 and 6.

---

## 2. Restricted data — NOT redistributed

### 2.1 Competition specification

The station size tiers, per-tier construction cost and daily capacity, the
`S1` / `S2` satisfaction ladders, and the objective weights `W1, W2, W3` come
from the specification documents of the source modelling competition. Those
documents are under a competition agreement and are **not** redistributed here.

* **Availability:** from the corresponding author on reasonable request.
* **What is shipped instead:** the *values themselves* are recorded verbatim in
  `code/utils/constants.py`, so every equation in the paper can be checked
  without the restricted documents. Nothing in the code reads the restricted
  files at run time; the constants are the single source of truth.

### 2.2 Artefacts derived from the competition specification

The following are **not** included, because they are derived from the
restricted specification and the author has not obtained permission to
redistribute derived results:

| Not shipped | Regenerate with |
|---|---|
| The list of 15,207 enumerated configurations | `code/02_core_solver/instance_generator.py` (`enumerate_configs()`) |
| Per-configuration enumeration output (`results_enumeration.json`) | `code/03_experiments/exp2_enumeration.py` |
| Per-configuration PoA / channel decomposition | same script (writes per-config channels directly) |
| `delta_S2_conditional_bound.json` | `code/03_experiments/exp2_layered_bounds.py` |

Both the *generator* and the *design* are shipped, so a reader can produce the
full result set locally. The design is fully specified and reproduces exactly:

```
10 candidate sites x 4 size tiers (cost 18 / 32 / 45)
total construction cost <= 118
minus the single all-zero (no station) assignment
= 15,207 valid configurations
```

`python code/02_core_solver/instance_generator.py` prints `15207`.

---

## 3. Synthetic data

The enumeration of §6 uses a synthetic 10-community instance. It is generated
from a fixed seed (`SEED_GEOMETRY = 2024` in
`code/02_core_solver/instance_generator.py`); no external data is involved and
the generation is fully deterministic.

One geometric knob is calibrated rather than measured — the side length of the
synthetic city, which controls how much the service areas of neighbouring
stations overlap and hence how much strategic congestion the decentralized
equilibrium generates. See `docs/REPRODUCIBILITY.md`, *Calibration of the
synthetic geometry*, for the calibration table and its effect on the reported
statistics.

---

## 4. Provenance of the shipped reference outputs

`results/figures/` and `results/tables/` contain the published figures and the
bodies of the published tables. They are shipped as **reference copies for
cross-checking** — the values in them are already public in the paper — and are
overwritten when the pipeline is re-run. They are not a substitute for the
archived raw result files listed in §2.2.
