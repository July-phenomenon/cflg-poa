# data/processed

Derived, redistribution-safe inputs. These are the only data files needed to
reproduce experiment 1 (§5), Figures 1 and 4, and Tables 2–4 and 6.

| File | Rows | Contents |
|---|---|---|
| `21city_parameters.json` | 21 | Per-city instance parameters: number of communities `n_com`, cluster radius `r_c`, number of candidate stations `n_sta`, population and area |
| `21city_parameters.csv` | 21 | Same content in tabular form |
| `21city_summary_statistics.csv` | — | Cross-city summary statistics quoted in §5.1 |

Provenance: derived from the publicly available accessibility rasters of
Han et al. (2026), Zenodo DOI `10.5281/zenodo.17010522` (CC BY 4.0), by the
scripts in `code/01_parameter_derivation/`.

Licence: CC BY 4.0.

Note that `enumeration_configs.csv` is deliberately **absent**: the 15,207
configuration list is derived from the restricted competition cost table, so
only the generator is shipped. Run

```
python code/02_core_solver/instance_generator.py
```

to obtain it (prints `15207`).
