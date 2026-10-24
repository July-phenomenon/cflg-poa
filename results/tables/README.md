# Published reference tables and figure inputs

This directory holds the **published reference copies** referenced by `.gitignore`
("only the published reference copies under `results/figures` and `results/tables`
are tracked"). Provenance of the files that matter for the submitted manuscript:

## Authoritative for the submitted figures

| File | Content | Rebuilt by |
|------|---------|-----------|
| `figure_inputs/Table5_2_21cities_PoA.csv` | `city, PoA` — coverage-locked specification, 21 cities (mean 1.0376, max 1.0791) | `code/05_figures_tables/export_exact_figure_inputs.py` |
| `figure_inputs/Table7_1_subsidy_comparison.csv` | `city, PoA_base, PoA_subsidized_consistent` — classic specification baseline and subsidy (mean 1.0503 -> 1.0656) | same script |

These two CSVs are 21-row aggregates of results already printed in the paper's
Tables 5 and 10, so shipping them does not widen the redistribution boundary.
They are consumed by `make_figures.py` (Figures 1 and 4) and
`make_figures_paper.py` (Graphical Abstract). Both scripts resolve them relative
to `results/tables/figure_inputs/`.

Upstream (private, regenerated locally, not redistributed):
`results/main/21cities_exact_locked.json` and
`results/main/_subsidy_exact/subsidy_exact.json`.

## Legacy — do not use as a source of truth for the manuscript

`Table5_2_21cities_PoA.csv` and `Table7_1_subsidy_comparison.csv` at the top level
of this directory come from an **earlier baseline** (36-wan station budget, before
the geometry unification) and disagree with the submitted manuscript. They are
kept only because `make_tables.py` writes to those paths and the table generators
still reference them. The difficult-to-regenerate reason: `make_tables.py` reads
`results/main/21城市*.json`, artefacts of that earlier run that are no longer
available. Prefer the `figure_inputs/` copies in any cross-check.
