# One-shot reproduction driver (Windows).
# Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1
#
# Any step can be re-run individually; see docs/REPRODUCIBILITY.md.
$ErrorActionPreference = "Stop"

$PY   = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$ROOT = Split-Path -Parent $PSScriptRoot

function Step($msg) { Write-Host ""; Write-Host "=== $msg ===" -ForegroundColor Cyan }
function Run($script, $extra = @()) {
    Write-Host "> $PY $script $extra"
    & $PY (Join-Path $ROOT $script) @extra
    if ($LASTEXITCODE -ne 0) { throw "$script failed with exit code $LASTEXITCODE" }
}

Set-Location $ROOT

Step "0/6  sanity: solver backend"
& $PY -c "import pulp; s=pulp.PULP_CBC_CMD(msg=0); assert s.available(), 'CBC missing'; print('CBC ok:', s)"
if ($LASTEXITCODE -ne 0) { throw "CBC solver not available - see docs/TROUBLESHOOTING.md" }

Step "0/6  sanity: enumeration design"
Run "code/02_core_solver/instance_generator.py"

Step "1/6  experiment 1 - 21 real cities (section 5)"
Run "code/03_experiments/exp1_21cities.py"
Run "code/03_experiments/exp1_regression.py"
Run "code/03_experiments/exp1_robustness.py"

Step "2/6  experiment 2 - full enumeration (section 6)"
$workers = if ($env:WORKERS) { $env:WORKERS } else { "8" }
Run "code/03_experiments/exp2_enumeration.py" @("--workers", $workers)
Run "code/03_experiments/exp2_channel_decomp.py"
Run "code/03_experiments/exp2_layered_bounds.py"
Run "code/03_experiments/exp2_connectivity_reg.py"

Step "3/6  experiment 3 - subsidy (section 7)"
Run "code/03_experiments/exp3_subsidy.py"
Run "code/03_experiments/exp3_policy_analysis.py"

Step "4/6  sensitivity analyses"
Run "code/04_sensitivity/capacity_heterogeneity.py"
Run "code/04_sensitivity/budget_sensitivity.py"
Run "code/04_sensitivity/cluster_radius.py"
Run "code/04_sensitivity/gamma_sensitivity.py"
Run "code/04_sensitivity/s2_robustness.py"

Step "5/6  figures"
Run "code/05_figures_tables/fig1_poa21.py"
Run "code/05_figures_tables/fig2_poa_dist.py"
Run "code/05_figures_tables/fig3_channel.py"
Run "code/05_figures_tables/fig4_subsidy.py"
Run "code/05_figures_tables/fig_ga.py"

Step "6/6  tables"
Run "code/05_figures_tables/make_tables.py"

Step "done"
Write-Host "Figures -> results/figures/"
Write-Host "Tables  -> results/tables/"
Write-Host "Logs    -> results/logs/"
