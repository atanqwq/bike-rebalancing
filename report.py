# -*- coding: utf-8 -*-
"""report.py —— L3 评估层与出图（M5 负责，对应课程 W9 可视化 + IEEE 制图规范）

约定
----
1. 所有图从 results/*.csv 读数据再画，**禁止手抄数字**（R3 可复现铁律）。
2. 尺寸按 IEEE：单栏 3.5 in、双栏 7.16 in，DPI=300，同时导出 PDF 与 PNG。
3. 图内一律用英文标注（论文正文是英文，避免中文字体缺失导致方块）。

产出
----
figures/fig1_architecture.(pdf|png)   Fig.1 系统架构（M1 主责，这里给可用初稿）
figures/fig2_clusters.(pdf|png)       Fig.2 站点分群 + k 选择
figures/fig3_forecast.(pdf|png)       Fig.3 预测效果
figures/fig4_convergence.(pdf|png)    Fig.4 GA/SA/Tabu 收敛曲线
figures/fig5_sensitivity.(pdf|png)    Fig.5 预测误差 → 调度成本
figures/fig6_scalability.(pdf|png)    Fig.6 规模可扩展性
tables/tab1_forecast.csv              Tab.1 预测模型对比
tables/tab2_ablation.csv              Tab.2 特征消融
tables/tab3_optimize.csv              Tab.3 优化算法对比
tables/tab4_literature_TEMPLATE.csv   Tab.4 文献对比表（M6 填写）
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import config as C

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": C.DPI,
    "savefig.bbox": "tight",
})


def _save(fig, name: str):
    C.ensure_dirs()
    for ext in ("pdf", "png"):
        fig.savefig(C.FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"[fig] {name}.pdf / .png")


def _read(path, **kw):
    p = C.RES_DIR / path
    if not p.exists():
        print(f"[warn] 缺少 {p}，跳过该图")
        return None
    return pd.read_csv(p, **kw)


# --------------------------------------------------------------------------
# Fig.1 架构图
# --------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=C.FIG_DOUBLE)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    layers = [
        ("L0  Data layer", "clean / features / normalization   (W9 data preprocessing)", 4.15),
        ("L1  Forecast layer", "Linear  ·  Random Forest  ·  FFBPN   (W5b, W7)", 3.05),
        ("L2  Optimization layer", "A* cost matrix (W2-3)  ·  CSP constraints (W4)  ·  GA/SA/Tabu (W5a)", 1.95),
        ("L3  Evaluation layer", "RMSE / R2  ·  cost  ·  sensitivity  ·  scalability", 0.85),
    ]
    colors = ["#DCE6F1", "#E2EFD9", "#FFF2CC", "#F2DCDB"]
    for (title, sub, y), col in zip(layers, colors):
        box = FancyBboxPatch((0.4, y), 9.2, 0.8, boxstyle="round,pad=0.02,rounding_size=0.08",
                             linewidth=1.0, edgecolor="#404040", facecolor=col)
        ax.add_patch(box)
        ax.text(0.6, y + 0.55, title, fontsize=9, fontweight="bold", va="center")
        ax.text(2.9, y + 0.55, sub, fontsize=7.5, va="center", color="#303030")

    for y0, y1 in [(4.15, 3.85), (3.05, 2.75), (1.95, 1.65)]:
        ax.add_patch(FancyArrowPatch((5.0, y0), (5.0, y1), arrowstyle="-|>",
                                     mutation_scale=12, linewidth=1.0, color="#404040"))
    ax.text(0.6, 0.30, "final deliverable:  Fig. 2-6 + Tab. 1-4  (paper Sec. 5)",
            fontsize=7, color="#606060")
    ax.set_title("Fig. 1  Prediction-Optimization framework for bike rebalancing", fontsize=9, pad=8)
    _save(fig, "fig1_architecture")


# --------------------------------------------------------------------------
# Fig.2 聚类
# --------------------------------------------------------------------------
def fig_clusters():
    st_path = C.DATA_DIR / "stations.csv"
    if not st_path.exists():
        print("[warn] 缺少 stations.csv，跳过 Fig.2")
        return
    st = pd.read_csv(st_path)
    ks = _read("E1_k_selection.csv")

    fig, axes = plt.subplots(1, 2, figsize=C.FIG_DOUBLE,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    cmap = plt.get_cmap("tab10")
    for k, g in st.groupby("cluster", observed=True):
        ax.scatter(g["x"], g["y"], s=28 + 90 * (g["base_demand"] / st["base_demand"].max()),
                   color=cmap(int(k) % 10), label=f"cluster {k}", alpha=0.85,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel("grid x (routing units)")
    ax.set_ylabel("grid y (routing units)")
    ax.set_title("(a) Station clusters on the routing grid", fontsize=8.5)
    ax.legend(loc="best", frameon=False)
    ax.set_aspect("equal", adjustable="datalim")

    ax2 = axes[1]
    if ks is not None:
        ax2.plot(ks["k"], ks["silhouette"], marker="o", markersize=4, linewidth=1.2)
        best = ks.loc[ks["silhouette"].idxmax()]
        ax2.scatter([best["k"]], [best["silhouette"]], s=60, facecolor="none",
                    edgecolor="red", linewidth=1.2, zorder=5)
        ax2.annotate(f"k*={int(best['k'])}", (best["k"], best["silhouette"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=7, color="red")
        ax2.set_xlabel("number of clusters k")
        ax2.set_ylabel("silhouette score")
        ax2.set_title("(b) k selection", fontsize=8.5)
    _save(fig, "fig2_clusters")


# --------------------------------------------------------------------------
# Fig.3 预测效果
# --------------------------------------------------------------------------
def fig_forecast():
    p = C.DATA_DIR / "predictions.csv"
    if not p.exists():
        print("[warn] 缺少 predictions.csv，跳过 Fig.3")
        return
    pr = pd.read_csv(p, parse_dates=["datetime"])

    fig, axes = plt.subplots(1, 2, figsize=C.FIG_DOUBLE)
    ax = axes[0]
    sid = int(pr.groupby("station_id")["y_true"].mean().idxmax())
    one = pr[pr["station_id"] == sid].sort_values("datetime").head(24 * 7)
    ax.plot(one["datetime"], one["y_true"], label="actual", linewidth=1.1, color="black")
    if "y_pred_rf" in one:
        ax.plot(one["datetime"], one["y_pred_rf"], label="RF", linewidth=1.1,
                linestyle="--", color="#C00000")
    if "y_pred_ffbpn" in one:
        ax.plot(one["datetime"], one["y_pred_ffbpn"], label="FFBPN", linewidth=1.0,
                linestyle=":", color="#2E75B6")
    ax.set_xlabel("time (one week window)")
    ax.set_ylabel("demand (bikes/hour)")
    ax.set_title(f"(a) Station {sid}: prediction vs actual", fontsize=8.5)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.tick_params(axis="x", rotation=20)

    ax2 = axes[1]
    if "y_pred_rf" in pr:
        ax2.scatter(pr["y_true"], pr["y_pred_rf"], s=4, alpha=0.25, color="#2E75B6",
                    edgecolors="none")
        lim = [0, max(pr["y_true"].max(), pr["y_pred_rf"].max()) * 1.05]
        ax2.plot(lim, lim, "k--", linewidth=0.8, label="y = x")
        ax2.set_xlim(lim)
        ax2.set_ylim(lim)
        ax2.set_xlabel("actual demand")
        ax2.set_ylabel("predicted demand (RF)")
        ax2.set_title("(b) Test-segment accuracy", fontsize=8.5)
        ax2.legend(frameon=False, loc="upper left")
    _save(fig, "fig3_forecast")


# --------------------------------------------------------------------------
# Fig.4 收敛曲线
# --------------------------------------------------------------------------
def fig_convergence(traces: dict | None = None, from_csv: bool = False):
    """traces: {algo: [hist, ...]}；也可从 results/E5_traces.npz 读取。"""
    if traces is None:
        npz = C.RES_DIR / "E5_traces.npz"
        if not npz.exists():
            print("[warn] 缺少 E5_traces.npz，跳过 Fig.4")
            return
        data = np.load(npz, allow_pickle=True)
        traces = {k: [np.asarray(h, dtype=float) for h in data[k]] for k in data.files}

    fig, ax = plt.subplots(figsize=C.FIG_SINGLE)
    styles = {"ga": ("#C00000", "-", "GA"), "sa": ("#2E75B6", "--", "SA"),
              "tabu": ("#548235", "-.", "Tabu")}
    budget = C.OPT_BUDGET
    for algo, hists in traces.items():
        if algo == "greedy" or not len(hists):
            continue
        # 三种算法的轨迹长度不同（GA 按代记录、SA/Tabu 按迭代记录），
        # 统一重采样到同一条「评估次数」轴上再比较，否则会被最短的那条截断。
        hists = [np.asarray(h, dtype=float) for h in hists if len(h) > 1]
        if not hists:
            continue
        L = max(len(h) for h in hists)
        grid = np.linspace(0, 1, L)
        arr = np.vstack([np.interp(grid, np.linspace(0, 1, len(h)), h) for h in hists])
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        x = np.linspace(0, budget, L)
        col, ls, lab = styles.get(algo, ("gray", "-", algo))
        ax.plot(x, mean, color=col, linestyle=ls, linewidth=1.2, label=lab)
        ax.fill_between(x, mean - std, mean + std, color=col, alpha=0.15, linewidth=0)
    ax.set_xlabel("evaluation budget (equal for all algorithms)")
    ax.set_ylabel("best feasible cost")
    ax.set_title("Convergence of metaheuristics", fontsize=9)
    ax.legend(frameon=False)
    _save(fig, "fig4_convergence")


# --------------------------------------------------------------------------
# Fig.5 敏感性
# --------------------------------------------------------------------------
def fig_sensitivity():
    e6 = _read("E6_sensitivity.csv")
    if e6 is None:
        return
    fig, ax = plt.subplots(figsize=C.FIG_SINGLE)
    x = e6["noise_level"] * 100
    if "regret_pct" in e6:
        y, yerr, ylab = e6["regret_pct"], e6.get("regret_std"), "cost penalty vs. perfect forecast (%)"
    else:
        y, yerr, ylab = e6["cost_increase_pct"], e6.get("cost_std"), "rebalancing cost increase (%)"
    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, marker="o", markersize=4, linewidth=1.3,
                    capsize=3, color="#C00000")
    else:
        ax.plot(x, y, marker="o", markersize=5, linewidth=1.3, color="#C00000")
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:+.1f}%", (xi, yi), textcoords="offset points",
                    xytext=(4, 7), fontsize=7)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_xlabel("forecast error injected (%)")
    ax.set_ylabel(ylab)
    ax.set_title("Coupling effect: forecast error to dispatch cost", fontsize=9)
    _save(fig, "fig5_sensitivity")


# --------------------------------------------------------------------------
# Fig.6 可扩展性
# --------------------------------------------------------------------------
def fig_scalability():
    e7 = _read("E7_scale.csv")
    if e7 is None:
        return
    fig, ax = plt.subplots(figsize=C.FIG_SINGLE)
    x = np.arange(len(e7))
    w = 0.36
    ax.bar(x - w / 2, e7["astar_time"], width=w, label="A* cost matrix", color="#8FAADC")
    ax.bar(x + w / 2, e7["opt_time"], width=w, label="GA optimization", color="#F4B183")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(n)}" for n in e7["n_stations"]])
    ax.set_xlabel("number of stations")
    ax.set_ylabel("runtime (s)")
    ax.set_title("Scalability of routing and optimization", fontsize=9)
    ax.legend(frameon=False)
    _save(fig, "fig6_scalability")


# --------------------------------------------------------------------------
# 表格
# --------------------------------------------------------------------------
def write_tables():
    C.ensure_dirs()
    e2, e3 = _read("E2_baseline.csv"), _read("E3_main.csv")
    if e2 is not None:
        both = pd.concat([x for x in (e2, e3) if x is not None], ignore_index=True)
        tab1 = (both.groupby("model", as_index=False)
                .agg(rmse=("rmse", "mean"), mae=("mae", "mean"), r2=("r2", "mean"))
                .round(4).sort_values("r2", ascending=False))
        tab1.to_csv(C.TAB_DIR / "tab1_forecast.csv", index=False)
        print("\n[Tab.1]\n", tab1.to_string(index=False))

    e4 = _read("E4_ablation.csv")
    if e4 is not None:
        e4.to_csv(C.TAB_DIR / "tab2_ablation.csv", index=False)
        print("\n[Tab.2]\n", e4.to_string(index=False))

    e5 = _read("E5_compare.csv")
    if e5 is not None:
        tab3 = (e5.groupby("algo", as_index=False)
                .agg(cost_mean=("final_cost", "mean"), cost_std=("final_cost", "std"),
                     travel_mean=("travel", "mean"), unmet_mean=("unmet", "mean"),
                     time_mean=("time", "mean")).round(3))
        tab3.to_csv(C.TAB_DIR / "tab3_optimize.csv", index=False)
        print("\n[Tab.3]\n", tab3.to_string(index=False))

    tpl = C.TAB_DIR / "tab4_literature_TEMPLATE.csv"
    if not tpl.exists():
        pd.DataFrame([
            {"ref": "[1]", "method": "Random Forest", "data": "UCI Bike Sharing",
             "metric": "RMSE 42.3", "limitation": "no rebalancing stage"},
            {"ref": "[2]", "method": "GA for VRP", "data": "synthetic 50 nodes",
             "metric": "cost -12%", "limitation": "static demand"},
        ]).to_csv(tpl, index=False)
    print("\n[Tab.4] 模板已就绪（M6 按文献填写）:", tpl.name)


def run_allplots(traces: dict | None = None):
    fig_architecture()
    fig_clusters()
    fig_forecast()
    fig_convergence(traces)
    fig_sensitivity()
    fig_scalability()
    write_tables()


if __name__ == "__main__":
    run_allplots()
    print("\n[自测] 图目录内容:")
    for f in sorted(C.FIG_DIR.glob("*")):
        print("  ", f.name, f"{f.stat().st_size/1024:.0f} KB")
