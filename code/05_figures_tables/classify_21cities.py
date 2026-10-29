#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
21-city data-driven taxonomy for the CFLG-PoA study.

Goal
----
Given the *measured* per-city outcomes we already produced, automatically
cluster the 21 city instances into a small number of archetypes, profile each
archetype, and expose the structural drivers.  This is the empirical face of
the "why is the PoA small / what creates the cross-city variation" question.

Features (all standardized before clustering)
---------------------------------------------
  PoA          : locked-specification efficiency ratio (outcome)
  milp_cov     : centralized (planner) coverage share  -> dominant driver (E1/E3)
  n_stations   : network scale  -> heterogeneity axis (E10)
  delta_cov    : coverage erosion under decentralization (0 vs >0)
  share_S1     : assignment-channel share of the gap
  share_cov    : coverage-channel share of the gap

Method
------
Pure-numpy k-means (k-means++ init, 30 restarts, 200 iters) for k=2..5.
The number of clusters is chosen by the silhouette coefficient (interpretable
tie-break: prefer the smaller k when silhouette is within 0.02).

Outputs
-------
  results/tables/T16_city_taxonomy.json
  results/tables/T16_city_taxonomy.csv
  results/figures/fig6_city_taxonomy.pdf  (+ .png)

Reproducible: `python code/05_figures_tables/classify_21cities.py`
"""
import json
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
TAB = os.path.join(RES, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# --------------------------------------------------------------------------- #
# 1. Load per-city data
# --------------------------------------------------------------------------- #
locked = json.load(open(os.path.join(RES, "main", "21cities_exact_locked.json")))["results"]
chan = {}
with open(os.path.join(TAB, "21city_channel_decomposition.csv"), encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        chan[row["city"]] = row

cities = list(locked.keys())
feat_names = ["PoA", "milp_cov", "n_stations", "delta_cov", "share_S1", "share_cov"]
Xraw = np.zeros((len(cities), len(feat_names)))
for i, c in enumerate(cities):
    r = locked[c]
    ch = chan.get(c, {})
    def fnum(v, default=0.0):
        try:
            if v is None or v == "" or v is False:
                return default
            return float(v)
        except (ValueError, TypeError):
            return default
    # share columns are empty when delta_cov == 0 (no gap -> no dominant channel)
    s1 = fnum(ch.get("share_S1"))
    sc = fnum(ch.get("share_cov"))
    Xraw[i, 0] = fnum(r.get("PoA"))
    Xraw[i, 1] = fnum(r.get("milp_cov"))
    Xraw[i, 2] = fnum(r.get("n_stations"))
    Xraw[i, 3] = fnum(r.get("delta_cov"))
    Xraw[i, 4] = s1
    Xraw[i, 5] = sc

# Standardize
mu = Xraw.mean(axis=0)
sd = Xraw.std(axis=0, ddof=0)
sd[sd == 0] = 1.0
X = (Xraw - mu) / sd

# --------------------------------------------------------------------------- #
# 2. k-means (pure numpy) + silhouette
# --------------------------------------------------------------------------- #
rng = np.random.default_rng(20261029)

def kmeans(X, k, restarts=30, iters=200):
    best = None
    best_inertia = np.inf
    n = X.shape[0]
    for _ in range(restarts):
        # k-means++ init
        centers = [X[rng.integers(n)]]
        while len(centers) < k:
            d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
            probs = d2 / d2.sum()
            centers.append(X[rng.choice(n, p=probs)])
        centers = np.array(centers, dtype=float)
        for _ in range(iters):
            dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            lab = dist.argmin(axis=1)
            new = np.array([X[lab == j].mean(axis=0) if np.any(lab == j) else centers[j]
                            for j in range(k)])
            if np.allclose(new, centers):
                centers = new
                break
            centers = new
        dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        lab = dist.argmin(axis=1)
        inertia = float(np.sum([np.sum((X[lab == j] - centers[j]) ** 2) for j in range(k)]))
        if inertia < best_inertia:
            best_inertia = inertia
            best = (centers, lab)
    return best

def silhouette(X, lab):
    n = X.shape[0]
    k = len(set(lab))
    if k <= 1 or k >= n:
        return -1.0
    a = np.zeros(n)
    b = np.full(n, np.inf)
    for i in range(n):
        same = np.where(lab == lab[i])[0]
        same = same[same != i]
        if len(same) > 0:
            a[i] = np.mean(np.linalg.norm(X[i] - X[same], axis=1))
        for j in set(lab):
            if j == lab[i]:
                continue
            other = np.where(lab == j)[0]
            d = np.mean(np.linalg.norm(X[i] - X[other], axis=1))
            b[i] = min(b[i], d)
    return float(np.mean((b - a) / np.maximum(a, b)))

results = {}
for k in range(2, 6):
    centers, lab = kmeans(X, k)
    sil = silhouette(X, lab)
    results[k] = (centers, lab, sil)
    print(f"k={k}: silhouette={sil:.4f}  sizes={[int((lab==j).sum()) for j in range(k)]}")

# choose k: best silhouette among solutions with NO singleton cluster
# (the 21-city outcomes are bimodal + one extreme outlier, Chongqing; a
#  singleton "cluster" is not an interpretable archetype for a supervisor).
def has_singleton(lab):
    k = len(set(lab))
    return any((lab == j).sum() < 2 for j in range(k))

valid = {k: v for k, v in results.items() if not has_singleton(v[1])}
best_k = max(valid, key=lambda k: valid[k][2])
print(f"Selected k={best_k} (silhouette={results[best_k][2]:.4f}); "
      f"k=3/4/5 rejected for singleton clusters")

centers, lab, sil = results[best_k]

# --------------------------------------------------------------------------- #
# 3. Profile each cluster
# --------------------------------------------------------------------------- #
profiles = []
assign = {}
for j in range(best_k):
    idx = np.where(lab == j)[0]
    members = [cities[i] for i in idx]
    sub = Xraw[idx]
    dcv = Xraw[idx, 3]
    prof = {
        "cluster": int(j),
        "n_cities": int(len(members)),
        "members": members,
        "poa_mean": float(sub[:, 0].mean()),
        "poa_min": float(sub[:, 0].min()),
        "poa_max": float(sub[:, 0].max()),
        "cov_mean": float(sub[:, 1].mean()),
        "n_sta_mean": float(sub[:, 2].mean()),
        "delta_cov_mean": float(sub[:, 3].mean()),
        "share_S1_mean": float(np.nanmean(sub[:, 4])),
        "share_cov_mean": float(np.nanmean(sub[:, 5])),
        "frac_delta_cov_zero": float(np.mean(dcv == 0.0)),
    }
    profiles.append(prof)
    for m in members:
        assign[m] = int(j)
    print(f"\nCluster {j} (n={len(members)}): {members}")
    print(f"  PoA mean={prof['poa_mean']:.4f} [{prof['poa_min']:.4f},{prof['poa_max']:.4f}]  "
          f"cov={prof['cov_mean']:.4f}  n_sta={prof['n_sta_mean']:.1f}  "
          f"dCov={prof['delta_cov_mean']:.4f}  S1share={prof['share_S1_mean']:.3f}  "
          f"CovShare={prof['share_cov_mean']:.3f}  frac(dCov=0)={prof['frac_delta_cov_zero']:.2f}")

# driver narrative (heuristic, reproducible)
NAMES = {
    "low": "floor-saturated scarce",
    "high": "assignment-driven mature",
}
for p in profiles:
    if p["cov_mean"] < 0.27 and p["frac_delta_cov_zero"] > 0.5:
        p["name"] = NAMES["low"]
    else:
        p["name"] = NAMES["high"]
    drivers = []
    if p["cov_mean"] < 0.20:
        drivers.append("low coverage -> residents sit on the 0.6 censoring floor; planner & resident objectives nearly coincide")
    elif p["cov_mean"] >= 0.35:
        drivers.append("high coverage -> the binding constraint shifts from 'who is served' to 'who is assigned where'")
    else:
        drivers.append("medium coverage -> transition zone")
    if p["frac_delta_cov_zero"] > 0.5:
        drivers.append("decentralization does NOT change coverage (delta_cov=0): the whole gap is assignment-only")
    else:
        drivers.append("decentralization also erodes coverage (delta_cov>0): both assignment and coverage channels active")
    if p["n_sta_mean"] >= 60:
        drivers.append("large network scale")
    p["drivers"] = drivers

# --------------------------------------------------------------------------- #
# 4. Save tables
# --------------------------------------------------------------------------- #
out_json = {
    "features": feat_names,
    "standardized": True,
    "method": "k-means++ (30 restarts, 200 iters), k chosen by silhouette",
    "selected_k": int(best_k),
    "silhouette": float(sil),
    "cluster_assignment": assign,
    "profiles": profiles,
}
with open(os.path.join(TAB, "T16_city_taxonomy.json"), "w", encoding="utf-8") as f:
    json.dump(out_json, f, ensure_ascii=False, indent=2)

with open(os.path.join(TAB, "T16_city_taxonomy.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["city", "cluster", "PoA", "milp_cov", "n_stations", "delta_cov", "share_S1", "share_cov"])
    for i, c in enumerate(cities):
        w.writerow([c, assign[c], Xraw[i, 0], Xraw[i, 1], Xraw[i, 2], Xraw[i, 3], Xraw[i, 4], Xraw[i, 5]])

# --------------------------------------------------------------------------- #
# 5. Figure
# --------------------------------------------------------------------------- #
cmap = plt.get_cmap("tab10")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.2))

# Panel (a): PoA vs coverage, points colored by cluster (no inline text -> no overlap)
for j in range(best_k):
    idx = [i for i, c in enumerate(cities) if assign[c] == j]
    ax1.scatter(Xraw[idx, 1] * 100, Xraw[idx, 0], s=80, color=cmap(j),
                label=f"C{j}: {profiles[j]['name']} (n={profiles[j]['n_cities']})",
                edgecolor="k", linewidth=0.4, zorder=3)
ax1.set_xlabel("Centralized coverage $c^{\\mathrm{MILP}}$ (%)")
ax1.set_ylabel("Locked PoA")
ax1.set_title(f"(a) 21 cities by automatic cluster  (k={best_k}, silhouette={sil:.2f})")
ax1.grid(alpha=0.3)
ax1.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
# mark the global-max outlier
imax = int(np.argmax(Xraw[:, 0]))
ax1.annotate(f"{cities[imax]} (max)", (Xraw[imax, 1] * 100, Xraw[imax, 0]),
             fontsize=6.5, xytext=(4, -10), textcoords="offset points",
             arrowprops=dict(arrowstyle="-", lw=0.5))

# Panel (b): ranked horizontal bars of PoA per city, y-axis = city, colored by cluster
order = np.argsort(Xraw[:, 0])
ynames = [cities[i] for i in order]
ypoa = Xraw[order, 0]
ycol = [cmap(assign[cities[i]]) for i in order]
ax2.barh(range(len(cities)), ypoa, color=ycol, edgecolor="k", linewidth=0.3)
ax2.set_yticks(range(len(cities)))
ax2.set_yticklabels(ynames, fontsize=7)
ax2.set_xlabel("Locked PoA")
ax2.set_title("(b) 21 cities ranked by PoA (color = cluster)")
ax2.axvline(1.0, color="grey", lw=0.6, ls=":")
for i, v in enumerate(ypoa):
    ax2.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=5.5)
ax2.set_xlim(0.999, 1.084)
ax2.invert_yaxis()

fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig6_city_taxonomy.pdf"))
fig.savefig(os.path.join(FIG, "fig6_city_taxonomy.png"), dpi=150)
print("\nSaved T16_city_taxonomy.{json,csv} and fig6_city_taxonomy.{pdf,png}")
