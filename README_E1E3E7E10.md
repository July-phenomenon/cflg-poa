# E1 / E3 / E7 / E10 实验脚本与产物索引（复现归类）

> 用途：把“规划对账表 · 二、部分完成”四个实验（E1 21城分类、E3 覆盖–PoA 单调、E7 参数敏感性、E10 跨城异质性）的**入口脚本 → 数据产物 → 稿件落点**一次性列清，方便任何人从零复现。
> 复现环境：Windows + `D:/Python39/python.exe`（numpy 1.26 / matplotlib 3.9 / scipy 1.13 / statsmodels 0.14）。
> 所有脚本默认从仓库根读取（`CFLG_ROOT` 缺省为脚本上两级目录），在仓库根执行即可。

---

## E1 — 21 城按覆盖分类与地板机制（§5.6 / Table 4 = `tab:covbands`）

| 项 | 内容 |
|---|---|
| 入口脚本 | `code/05_figures_tables/table_cov_bands.py` |
| 输入 | `results/main/21cities_exact_locked.json`（21 城 locked 规格主结果） |
| 产物 | `results/tables/T13_coverage_bands.{csv,json,tex}`（三档组内 PoA 均值/中位/极值、覆盖均值、Δcov、cut-point 敏感性） |
| 稿件落点 | §5.6 “Classification by coverage” 段 + Table 4（`\label{tab:covbands}`，line ~588） |
| 关键数字 | 低<20% (n=4) PoA 1.0074｜中20–35% (n=11) 1.0324｜高≥35% (n=6) 1.0674；单调成立；cut 移动 ±2pp 不变 |
| 复现验证 | 2026-10-29 重跑：输出与稿件逐字一致，monotone=True |

> 机制含义：低覆盖时约七成老人坐在满意度封底（floor=0.6），中央与居民目标几乎重合 → 效率损失最小；高覆盖城市（北京/成都/长沙/上海/武汉/重庆）损失最大，重庆 1.0791 封顶。

---

## E3 — 覆盖–PoA 单调关系专用图（Fig.2(a)）

| 项 | 内容 |
|---|---|
| 入口脚本 | `code/05_figures_tables/fig5_coverage_poa.py`（面板 a） |
| 输入 | `results/main/21cities_exact_locked.json` |
| 产物 | `results/figures/fig5_coverage_poa.{pdf,png}` |
| 稿件落点 | Fig.2（`\label{fig:covpoa}` 面板 a，line ~606） |
| 关键数字 | PoA vs 模型覆盖 c^MILP：r=+0.842 (p=1.7e-6)，线性拟合 R²=0.71 |
| 复现验证 | 2026-10-29 重跑：r=+0.842, R²=0.709, 与图注吻合 |

> 注意：Fig.2(a) 是 **模型自身覆盖** c^MILP，区别于 §5.6 外部验证 Fig.3 的“2025 真实护理型可达率”；两者方向一致，但口径不同，不可混为一数。

---

## E7 — 参数敏感性审计（附录 C = `tab:sensitivity` + `tab:sensaudit`）

| 项 | 内容 |
|---|---|
| 入口脚本 | `code/04_sensitivity/audit_sensitivity_provenance.py` |
| 输入 | `results/sensitivity/cap_floor_theta_0.{50,60,70}_exact_per_config.json`、`results/sensitivity/ladder_K{5,10,20,50,100}_per_config.json`、`results/main/cluster_radius_sensitivity_summary.json`、`results/tables/T11_enum_connectivity_reg.json`、`results/tables/T15_enum_regression_ladder.json` |
| 产物 | `results/tables/T14_sensitivity_audit.{json,csv,md}`（逐行 verdict） |
| 稿件落点 | 附录 C “Additional Sensitivity Analyses”（line ~1074）：θ 网格段 + Table `tab:sensitivity` + “Provenance” 段 + Table `tab:sensaudit` |
| 覆盖维度 | 满意度 floor θ、满意度阶梯 K、簇半径、容量档 A/B/C、单站预算 25/30/36/40 wan、连通性回归 |
| 复现验证 | 2026-10-29 重跑：18 项 REPRODUCED + 4 项 supplementary；唯一 “TEXT FIX NEEDED” 为枚举连通性 β=0.152，但**稿件从未以 0.152 出现**（只报嵌套 R²），属对旧草稿的误报，无需改稿 |

> 全局最大 PoA 不总超 1/θ：θ=0.50 时 1.8798<2.0；θ=0.60 时 1.6404<1.667；仅 θ=0.70 的 1.6017>1.429（因未锁定层在 Proposition 1 前提之外）。凡说“远超 1/θ”必须逐 grid 限定。

---

## E10 — 跨城异质性：网络规模而非地理（§5.6 回归段 + 附录 C 段 + Fig.2(b)）

| 项 | 内容 |
|---|---|
| 入口脚本 | `code/05_figures_tables/fig5_coverage_poa.py`（面板 b）+ `code/03_experiments/exp2_enum_regression_full.py` |
| 输入 | `21cities_exact_locked.json`（面板 b）；`results_enumeration_exact.json`（15,207 配置回归阶梯） |
| 产物 | `results/figures/fig5_coverage_poa.pdf`（面板 b）+ `results/tables/T15_enum_regression_ladder.{json,md}` |
| 稿件落点 | §5.6 回归段（log n_sta β=+0.0246, p=0.0023）+ 附录 C “Cross-city heterogeneity” 段（line ~1170）+ Fig.2(b) |
| 关键数字 | PoA vs log10(n_sta)：r=+0.730 (p=0.0002)；bottom-5 网络均值 1.0063 vs top-5 1.0611；嵌套 R²=0.0069/0.194/0.200/0.204 |
| 复现验证 | 2026-10-29 重跑：r=+0.730, R²=0.533, 异质性 1.0063/1.0611，阶梯 R² 与 `tab:sensaudit` 一致 |

> 报告口径：n=21、4 个预测变量，未做多重比较校正，稿件明确标注为 **exploratory**（探索性），并给出留一城重估 21/21 稳健作为支撑。

---

## 一键复现命令（仓库根）

```bash
export CFLG_ROOT="$(pwd)"   # Windows Git Bash 也可省略，脚本自动定位
D:/Python39/python.exe code/05_figures_tables/table_cov_bands.py
D:/Python39/python.exe code/05_figures_tables/fig5_coverage_poa.py
D:/Python39/python.exe code/04_sensitivity/audit_sensitivity_provenance.py
D:/Python39/python.exe code/03_experiments/exp2_enum_regression_full.py
```

四脚本均为**只读式复算**（不重解 MILP），秒级完成；输出与稿件数字逐项一致，可放心作为“数据说话”的审计证据。
