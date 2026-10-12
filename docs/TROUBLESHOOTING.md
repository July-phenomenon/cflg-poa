# Troubleshooting

## `FileNotFoundError: poa_full_per_config_s2.json`

Expected in a fresh checkout. The enumeration output is not redistributed
(see `data/external/DATA_SOURCES.md` §2.2). Either

```
python code/03_experiments/exp2_enumeration.py --workers 8
```

or point the lookup at a file you already have:

```
export CFLG_ENUM_JSON=/path/to/my_enumeration.json      # Linux/macOS
set CFLG_ENUM_JSON=C:\path\to\my_enumeration.json       # Windows cmd
```

Figures 1 and 4 do **not** need this file and will still build.

---

## `SystemExit: results/main/ is missing the following experiment outputs`

The table scripts refuse to run before the experiments have produced their
outputs. The message lists exactly which files are missing and which scripts
generate them. Run `scripts/run_all.sh`, or the specific experiment, first.

---

## The enumeration is very slow

* Use `--workers N` (start with the number of physical cores).
* Use `--limit N` for a smoke test (e.g. `--limit 200` finishes in ~2 min
  with 4 workers).
* Configurations with more built stations take longer; the sweep therefore
  speeds up and slows down as it proceeds, which is normal.

---

## `PuLP: solver not found` / `PULP_CBC_CMD unavailable`

PuLP ≥ 2.7 normally bundles CBC. If the bundled binary is missing (some
minimal Linux containers), install CBC system-wide:

```
conda install -c conda-forge coincbc      # or
apt-get install coinor-cbc
```

Then verify with the snippet in `docs/ENVIRONMENT.md`.

---

## Figures use the wrong font

The plotting template requests Arial and falls back to DejaVu Sans, which
matplotlib always has. If Arial is installed but not found, clear the
matplotlib font cache:

```
python -c "import matplotlib, shutil, os; \
shutil.rmtree(matplotlib.get_cachedir())"
```

Font choice affects glyph metrics only, not any number.

---

## Numbers differ in the last decimal from the paper

Possible legitimate causes:

1. **CBC version.** Different CBC builds can return a different optimal basis
   for degenerate LPs. The reported PoA is a *lower bound*, so any suboptimality
   in the centralized solve understates the gap — the direction is safe.
2. **Sampler version.** `numpy.random.default_rng` is stable across NumPy ≥ 1.17,
   but `RandomState` is not interchangeable with it. Do not substitute one for
   the other.
3. **The synthetic geometry knob** for §6 — see
   `docs/REPRODUCIBILITY.md` §4.2.

If a *sign* or an *ordering* differs (e.g. congestion no longer negative), that
is a real discrepancy: please open an issue with the command and the output.

---

## `ModuleNotFoundError: core_solver`

Scripts add their package directory to `sys.path` themselves. If you moved a
script out of the package, run it from the repository root instead:

```
python code/05_figures_tables/fig1_poa21.py      # correct
cd code/05_figures_tables && python fig1_poa21.py  # also works
```

---

## Everything writes to the wrong directory

Scripts resolve the repository root from `$CFLG_ROOT` and otherwise from the
script location. Setting `CFLG_ROOT` explicitly is the supported override:

```
export CFLG_ROOT=/path/to/cflg-poa-reproducibility
```
