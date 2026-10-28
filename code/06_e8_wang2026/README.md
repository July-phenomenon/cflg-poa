# E8 — External validation against independent 2025 data

This folder contains the scripts behind §5.6 (*External validation against
independent 2025 data*) of the manuscript. They answer one question:

> Do the paper's two headline empirical claims — **coverage is the binding
> constraint** and **the PoA rises with coverage** — survive on an
> *independent* 2024/25 dataset, rather than being artefacts of the single 2020
> calibration used in §5.1–5.5?

**Short answer: yes, directionally.** Model coverage and measured PoA both
correlate positively and significantly with the realised nursing-grade coverage
of Wang et al. (2026) across the 21 cities (r ≈ 0.53, p ≈ 0.01). The agreement
is *directional*, not a numerical identity (see caveats below).

---

## 1. Data

| Role | Dataset | DOI / access |
|---|---|---|
| 2020 calibration (paper baseline, **not** re-derived here) | Han et al. (2026), 21 cities, 100 m | Zenodo `10.5281/zenodo.17010522` |
| 2024/25 external validation | Wang et al. (2026), China, 1 km, affordability-constrained | Figshare `10.6084/m9.figshare.30478925` (CC BY 4.0) |
| City boundaries (only for the diagnostic parameter re-derivation) | DataV administrative boundaries | `https://geo.datav.aliyun.com/areas_v3/bound/<adcode>.json` |

The Wang2026 archive (`Elderly_Care_Accessibility_China_2025 v2.zip`, ~1.1 GB)
is **not** committed to this repository. Extract only what the scripts need:

```
1_Raster_Data/Baseline/Total_t60_k0.tif      # 1-km baseline accessibility
1_Raster_Data/Auxiliary/Population_60plus_2025.tif
2_Administrative_Aggregation/city_level.csv  # city-level accessible rates
2_Administrative_Aggregation/supply_demand_ratio.csv  # facility_count per city
4_Facility_Points/Facilities.csv
```

## 2. Scripts (run in this order)

| # | Script | Purpose | Key output |
|---|---|---|---|
| 0 | `derive_params_generic.py` | The **same three-step method** as `code/01_parameter_derivation/derive_21city_params.py`, generalised to clip a national raster by a polygon. Also used for the **self-check**. | one JSON record |
| 1 | *(self-check)* `derive_params_generic.py --raster <Han Beijing>` | Must reproduce the published Han parameters **exactly** (see §3). | — |
| 2 | `run_all_21.py` | Clip Wang2026 to 21 cities, re-derive `n_stations` / `cluster_r` (diagnostic comparability audit). | `results/e8_wang2026/w26_params_21city.*`, `compare_21city_w26_vs_han.csv` |
| 3 | `closed_loop_check.py` | Merge the paper's 21-city results with Wang2026 accessible rates; compute Pearson/Spearman. | `closed_loop_merged.csv`, `closed_loop_correlations.csv` |
| 4 | `make_e8_figure.py` | Figure 2 of the manuscript (`e8_external_validation`). | `results/figures/e8_external_validation.{png,pdf}` |

### Paths

Each script declares its input paths at the top and honours environment
overrides where noted:

```bash
export W26_DIR="D:/.../数据/Wang2026v2"      # run_all_21.py
export CFLG_ROOT="D:/.../cflg-poa-reproducibility"  # closed_loop_check.py, make_e8_figure.py
```

Run with the project's Python (needs `numpy pandas scipy rasterio matplotlib`):

```bash
python code/06_e8_wang2026/run_all_21.py
python code/06_e8_wang2026/closed_loop_check.py
python code/06_e8_wang2026/make_e8_figure.py
```

## 3. Self-check (mandatory before trusting anything)

The generalised extractor must reproduce the *published* Han parameters from
Han's own raster:

```
$ python derive_params_generic.py \
    --raster 数据/17010522/Accessibility/Accessibility/Beijing_Accessibility.tif \
    --high-pct 3.09 --city Beijing
# n_stations = 76      (published 76)
# n_communities = 345  (published 345)
# cluster_r_m = 3686.192128188809   (published 3686.192...)
# A_built_km2 = 2437.68 ; n_px_valid = 369648
```

If any of these differ, stop — the method or the environment has drifted.

## 4. Why the Wang2026 parameters are **not** swapped into the model

This is a deliberate, documented decision, not an omission:

1. **`n_communities` has no comparable layer.** Han's built-up area (Beijing
   2,437 km²) has no counterpart in Wang2026, whose 1-km raster covers all land
   (Beijing ≈ 11,622 km²). Re-deriving `n_communities` naively inflates it
   **4–10×**.
2. **`n_stations` is confounded by resolution.** At 1 km vs 100 m, peak
   detection finds a median **1.6×** more peaks; the difference cannot be
   separated from a genuine supply change.
3. **`facility_count` ≠ stations.** Wang2026 reports a facility census (Beijing
   795) that is a median **6.6×** the peak-based station proxy (76). They count
   different objects.

Therefore the 2025 data enter as an **external check on the coverage/PoA
ordering**, not as a re-calibration of the game instance. The 21-city numbers in
§5.1–5.5 are unchanged.

## 5. Caveats (also stated in the manuscript)

- Correlations are moderate (r ≈ 0.5) at n = 21; the level of significance is
  p ≈ 0.01, not stronger.
- Correlations with nursing-grade and all-bed rates are almost equal
  (0.53 vs 0.54): we do **not** claim "model coverage = nursing coverage", only
  that the model's coverage ordering tracks the realised supply structure.
- Two cities deviate: **Nanjing** (model 29.8 % vs nursing 92.5 %) and
  **Hangzhou** (model 33.6 % vs nursing 6.7 %).
- The two measures are different constructs (capacity-constrained assignment
  share vs threshold-based accessibility share).
