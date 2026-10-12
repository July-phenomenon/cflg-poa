# data/raw

This directory is intentionally almost empty.

## han_et_al_2026_rasters/

**Not included.** Download the 21-city accessibility rasters from

> Han et al. (2026), Zenodo, DOI `10.5281/zenodo.17010522` (CC BY 4.0)

and place them here. They are only needed if you want to re-run the parameter
derivation in `code/01_parameter_derivation/` from the rasters themselves.

If you only want to reproduce the results, this step is unnecessary: the
derived parameters are shipped in `../processed/21city_parameters.json`.

## competition_spec/

**Not included — restricted.** The competition specification documents are
subject to a competition agreement and are available from the corresponding
author on reasonable request.

Nothing in the code reads these files at run time. The parameter values they
define (capacity and cost tiers, satisfaction ladders, objective weights) are
recorded verbatim in `code/utils/constants.py`.
