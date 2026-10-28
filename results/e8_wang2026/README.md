# results/e8_wang2026 — outputs of the external-validation study (§5.6)

| File | Produced by | Content |
|---|---|---|
| `w26_params_21city.csv` / `.json` | `code/06_e8_wang2026/run_all_21.py` | Three-step parameters re-derived from the Wang2026 1-km raster for the 21 cities (diagnostic only — **not** fed into the model; see the code README). |
| `compare_21city_w26_vs_han.csv` | `run_all_21.py` | Side-by-side `n_stations` / `cluster_r` / `A_built` / `n_communities`, Han vs Wang2026, with ratios. |
| `closed_loop_merged.csv` | `closed_loop_check.py` | Per-city merge of the paper's 21-city results (`milp_cov`, `nash_cov`, `PoA`) and Wang2026 accessible rates (Total / Nursing, t60 k0). |
| `closed_loop_correlations.csv` | `closed_loop_check.py` | Pearson and Spearman correlations for the six headline comparisons (the numbers quoted in §5.6). |

The manuscript figure is `results/figures/e8_external_validation.{png,pdf}`.

## Headline numbers (reproduced by `closed_loop_check.py`)

| Comparison | Pearson r | Spearman ρ | p |
|---|---|---|---|
| model coverage vs real **nursing-grade** rate | +0.531 | +0.569 | 0.013 / 0.007 |
| model coverage vs real **all-bed** rate | +0.537 | +0.496 | 0.012 / 0.022 |
| **PoA** vs real nursing-grade rate | +0.530 | +0.553 | 0.013 / 0.009 |
| **PoA** vs real all-bed rate | +0.590 | +0.549 | 0.005 / 0.010 |
| PoA vs model's own coverage (internal) | +0.842 | +0.845 | <1e-5 |

Medians: model coverage **29.1 %**, real nursing-grade **17.85 %**, real all-bed **76.2 %**.
