# Environment

## Verified configuration

| Component | Value |
|---|---|
| OS | Windows 10/11 x64 (scripts are OS-agnostic; PowerShell and Bash drivers are both provided) |
| Python | 3.9.13 (verified); 3.10+ recommended |
| CPU | 8 cores recommended (the enumeration benefits from `--workers`) |
| RAM | 8 GB (the full enumeration holds ~15 k records, well under 1 GB) |
| Disk | < 100 MB for the package; ~2 GB if the rasters are downloaded |
| Network | **not required** — CBC ships with PuLP, no licence server is contacted |

## Dependencies

```
numpy>=1.24      scipy>=1.10     pandas>=2.0
matplotlib>=3.7  scikit-learn>=1.3   statsmodels>=0.14
pulp>=2.7
```

Exact versions last verified: see `requirements_frozen.txt`
(`matplotlib 3.9.4`, `numpy 1.26.4`, `pandas 2.3.3`, `pulp 3.3.1`,
`scikit-learn 1.6.1`, `scipy 1.13.1`, `statsmodels 0.14.6`).

## MILP backend

The mixed-integer programs are built with **PuLP** and solved with **CBC**
(COIN-OR Branch and Cut), which PuLP bundles as a binary — no separate
installation, no licence file, no registration.

To confirm the backend is usable:

```python
import pulp
p = pulp.LpProblem("t", pulp.LpMinimize); x = pulp.LpVariable("x", 0, 1)
p += x
print(pulp.PULP_CBC_CMD(msg=0).available())     # path to cbc.exe, or False
```

No commercial solver (Gurobi, CPLEX) is used anywhere, so there is no licence
barrier to reproduction.

## Optional: alternative backends

Any solver that PuLP can drive can be substituted without touching the model
code, e.g. `prob.solve(pulp.GLPK_CMD())`. Results may differ in the last digits
for degenerate instances; the reported PoA values are lower bounds, so a
suboptimal centralized solve can only *understate* the gap.

## Runtime

See the time budget in `README.md` §3. The dominant cost is experiment 2
(15,207 configurations × two solves each). Use `--workers` to parallelise:

```
python code/03_experiments/exp2_enumeration.py --workers 8
```
