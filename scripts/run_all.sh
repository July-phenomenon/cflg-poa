#!/usr/bin/env bash
# One-shot reproduction driver.
# Run from the repository root:   bash scripts/run_all.sh
#
# Any step can be re-run individually; see docs/REPRODUCIBILITY.md.
set -euo pipefail

PY="${PYTHON:-python}"

step() { printf '\n=== %s ===\n' "$1"; }

step "0/6  sanity: solver backend"
"$PY" - <<'PY'
import pulp
s = pulp.PULP_CBC_CMD(msg=0)
assert s.available(), "CBC solver not available - see docs/TROUBLESHOOTING.md"
print("CBC ok:", s)
PY

step "0/6  sanity: enumeration design"
"$PY" code/02_core_solver/instance_generator.py

step "1/6  experiment 1 - 21 real cities (section 5)"
"$PY" code/03_experiments/exp1_21cities.py
"$PY" code/03_experiments/exp1_regression.py
"$PY" code/03_experiments/exp1_robustness.py

step "2/6  experiment 2 - full enumeration (section 6)"
"$PY" code/03_experiments/exp2_enumeration.py --workers "${WORKERS:-8}"
"$PY" code/03_experiments/exp2_channel_decomp.py
"$PY" code/03_experiments/exp2_layered_bounds.py
"$PY" code/03_experiments/exp2_connectivity_reg.py

step "3/6  experiment 3 - subsidy (section 7)"
"$PY" code/03_experiments/exp3_subsidy.py
"$PY" code/03_experiments/exp3_policy_analysis.py

step "4/6  sensitivity analyses"
"$PY" code/04_sensitivity/capacity_heterogeneity.py
"$PY" code/04_sensitivity/budget_sensitivity.py
"$PY" code/04_sensitivity/cluster_radius.py
"$PY" code/04_sensitivity/gamma_sensitivity.py
"$PY" code/04_sensitivity/s2_robustness.py

step "5/6  figures"
"$PY" code/05_figures_tables/fig1_poa21.py
"$PY" code/05_figures_tables/fig2_poa_dist.py
"$PY" code/05_figures_tables/fig3_channel.py
"$PY" code/05_figures_tables/fig4_subsidy.py
"$PY" code/05_figures_tables/fig_ga.py

step "6/6  tables"
"$PY" code/05_figures_tables/make_tables.py

step "done"
echo "Figures -> results/figures/"
echo "Tables  -> results/tables/"
echo "Logs    -> results/logs/"
