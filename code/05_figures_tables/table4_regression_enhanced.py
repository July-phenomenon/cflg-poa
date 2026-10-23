# -*- coding: utf-8 -*-
"""
T10（P2-10）：21 城回归统计规范化增强
=====================================
在 exp1_regression.py（已发布，含 raw beta / 95% CI / VIF）基础上，补齐 1 区
审稿人通常要求的“效应量 + 多重共线诊断 + 样本稳健性”三件套：

  1. 标准化 β（standardized coefficient，即效应量）——所有变量 z-score 后
     重估，便于跨量纲比较各预测因子的相对影响力；
  2. 每个预测变量的 VIF（已在 exp1 算出 max=4.685，这里逐变量列出并复核）；
  3. Leave-one-city-out（LOO）稳健性——逐城剔除后重拟，统计
     log(n_sta) 系数显著性与整体模型 F 检验显著性的保持比例，以及系数区间。

输入：data/processed/21city_regression_input.csv（由 exp1_regression.py 产出的
      逐城派生表，含 PoA / connectivity / capacity_gini / n_stations / cluster_r）
输出：results/tables/T10_regression_enhanced.json
      （并打印可直接粘贴进 manuscript 的 LaTeX 行 + 稳健性结论文本）

复现：python table4_regression_enhanced.py
依赖：numpy, pandas, statsmodels（与 exp1_regression.py 同环境）
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "data", "processed", "21city_regression_input.csv")
OUT_DIR = os.path.join(ROOT, "results", "tables")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    df = pd.read_csv(SRC)
    N = len(df)
    assert N == 21, f"expected 21 cities, got {N}"

    df["log_n_stations"] = np.log(df["n_stations"])
    XCOLS = ["connectivity", "capacity_gini", "log_n_stations", "cluster_r"]
    y = df["PoA"].astype(float)
    X = sm.add_constant(df[XCOLS].astype(float))

    # ---------------------------------------------------------------- 1) raw OLS
    model = sm.OLS(y, X).fit()

    # ------------------------------------------------- 2) 标准化 β（效应量）
    # 预测变量与因变量均 z-score；标准化模型不含截距（均值中心化后截距为 0）
    z = lambda s: (s - s.mean()) / s.std(ddof=0)
    Xz = pd.DataFrame({c: z(df[c]) for c in XCOLS})
    yz = z(y)
    model_z = sm.OLS(yz, Xz).fit()  # 无 const

    # raw 与 standardized 的 p 值应一致（线性变换不改变显著性）
    std_beta = {c: float(model_z.params[c]) for c in XCOLS}

    # ----------------------------------------------------- 3) 逐变量 VIF
    X_vif = sm.add_constant(df[XCOLS].astype(float))
    vif_all = {X_vif.columns[i]: float(variance_inflation_factor(X_vif.values, i))
               for i in range(X_vif.shape[1])}
    vif_pred = {k: v for k, v in vif_all.items() if k != "const"}
    max_vif_pred = max(vif_pred.values())

    # --------------------------------------------- 4) Leave-one-city-out (LOO)
    loo = []
    for i, city in enumerate(df["city"]):
        mask = df["city"] != city
        Xi = sm.add_constant(df.loc[mask, XCOLS].astype(float))
        yi = df.loc[mask, "PoA"].astype(float)
        mi = sm.OLS(yi, Xi).fit()
        loo.append(dict(
            dropped=city,
            n=int(mask.sum()),
            log_n_stations_coef=float(mi.params["log_n_stations"]),
            log_n_stations_p=float(mi.pvalues["log_n_stations"]),
            log_n_stations_sig=bool(mi.pvalues["log_n_stations"] < 0.05),
            model_f_p=float(mi.f_pvalue),
            model_f_sig=bool(mi.f_pvalue < 0.05),
        ))
    n_sig_logn = sum(r["log_n_stations_sig"] for r in loo)
    n_sig_model = sum(r["model_f_sig"] for r in loo)
    coefs = [r["log_n_stations_coef"] for r in loo]
    ps = [r["log_n_stations_p"] for r in loo]

    # ----------------------------------------------------- 5) 汇总并写盘
    out = dict(
        n=N,
        dependent="PoA",
        regressors=XCOLS,
        spec="locked",
        raw_ols=dict(
            R2=float(model.rsquared),
            AdjR2=float(model.rsquared_adj),
            f_pvalue=float(model.f_pvalue),
            coef={n: float(model.params[n]) for n in X.columns},
            std_err={n: float(model.bse[n]) for n in X.columns},
            p_value={n: float(model.pvalues[n]) for n in X.columns},
            ci_low={n: float(model.conf_int().loc[n, 0]) for n in X.columns},
            ci_high={n: float(model.conf_int().loc[n, 1]) for n in X.columns},
        ),
        standardized_beta=std_beta,
        vif_all=vif_all,
        vif_predictors=vif_pred,
        max_vif_predictors=round(max_vif_pred, 6),
        loo=dict(
            n_cities=N,
            n_log_n_stations_significant=n_sig_logn,
            n_model_F_significant=n_sig_model,
            log_n_stations_coef_min=round(min(coefs), 6),
            log_n_stations_coef_max=round(max(coefs), 6),
            log_n_stations_p_min=round(min(ps), 6),
            log_n_stations_p_max=round(max(ps), 6),
            per_city=loo,
        ),
    )
    out_path = os.path.join(OUT_DIR, "T10_regression_enhanced.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # ----------------------------------------------------- 6) 打印可粘贴结果
    print("=== RAW OLS (locked) ===")
    print(f"R2={model.rsquared:.4f}  AdjR2={model.rsquared_adj:.4f}  F-p={model.f_pvalue:.4f}")
    print("\n=== Standardized beta (effect size) ===")
    for c in XCOLS:
        print(f"  {c:16s} std_beta={std_beta[c]:+.4f}  (p={model.pvalues[c]:.4f})")
    print("\n=== VIF (predictors) ===")
    for c in XCOLS:
        print(f"  {c:16s} VIF={vif_pred[c]:.3f}")
    print(f"  max predictor VIF = {max_vif_pred:.3f}")
    print("\n=== Leave-one-city-out (n=21) ===")
    print(f"  log(n_sta) significant in {n_sig_logn}/{N} drops "
          f"(coef range [{min(coefs):+.4f}, {max(coefs):+.4f}], "
          f"p range [{min(ps):.4f}, {max(ps):.4f}])")
    print(f"  model F-test significant in {n_sig_model}/{N} drops")
    print("\n[out]", out_path)

    # LaTeX 行（locked 列：raw beta / 95% CI / std beta）
    print("\n--- LaTeX rows (locked: beta [CI], std beta) ---")
    name_map = {"const": "const", "connectivity": "connectivity",
                "capacity_gini": "capacity\\_gini",
                "log_n_stations": "$\\log(n_{\\mathrm{sta}})$",
                "cluster_r": "cluster\\_r"}
    for n in X.columns:
        b = model.params[n]
        lo = model.conf_int().loc[n, 0]
        hi = model.conf_int().loc[n, 1]
        sb = std_beta.get(n, float("nan"))
        print(f"    {name_map.get(n, n)} & {b:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}] & {sb:+.4f} \\\\")


if __name__ == "__main__":
    main()
