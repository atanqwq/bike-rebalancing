# -*- coding: utf-8 -*-
"""optimize.py —— L2 优化层（M4 负责，对应课程 W4 CSP + W5a GA / SA / Tabu）

问题模型（论文 3.4 节必须原样写清，评审会追问这里）
----------------------------------------------------
一次「再平衡」：给定 N_VISIT 个候选站点，各自的**现有库存 inv** 与
**预测需求 target**，由 FLEET 辆车从「库存最多的站点（depot）」出发，
按某个访问顺序逐站装卸，最后评估两件事：

    travel = 每辆车的闭合回路行驶代价（来自 routing.py 的 A* 代价矩阵）
    unmet  = 访问结束后仍然缺少的车次数（缺口未被补上的部分）
    cost   = travel + LAMBDA_PENALTY × unmet        ← 唯一的目标函数

解编码：一个站点**排列**（permutation）。解码时按 FLEET 均分成若干段，
每段一辆车，装载量受 VEH_CAP 限制。因为是排列，交换与插入天然保持可行；
GA 的交叉必须用 **OX（顺序交叉）**，否则会出现非法解——这正是课件
Week5a「Problems with Crossover」那一页在讲的问题。

公平对比（E5 的关键）
--------------------
三个算法共用：同一初始解候选池、同一个 evaluate()、同一个评估次数预算
（C.OPT_BUDGET）。Tabu 的禁忌长度按课件公式 L = a×Nc + Random(g)，
其中 Nc 取当前「未满足车次总数」。

输出
----
results/E5_compare.csv    algo, seed, run, final_cost, travel, unmet, evals, time
results/E7_scale.csv      n_stations, astar_time, opt_time
results/E6_sensitivity.csv noise_level, final_cost, cost_increase_pct
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

import config as C


# --------------------------------------------------------------------------
# 上下文：把预测结果变成可优化的实例
# --------------------------------------------------------------------------
class Ctx:
    """一次再平衡实例的全部数据。"""

    def __init__(self, sub_cost, depot_pos, surplus, deficit, fleet, cap, lam,
                 q=None, truth_deficit=None, quota=None):
        self.sub = np.asarray(sub_cost, dtype=float)
        self.depot = int(depot_pos)
        self.surplus = np.asarray(surplus, dtype=float)
        self.deficit = np.asarray(deficit, dtype=float)
        self.fleet = int(fleet)
        self.cap = float(cap)
        self.lam = float(lam)
        self.n = len(self.surplus)
        # 一次出车最多服务的站点数：预测决定「先服务谁」，超出配额的站点本次不服务
        self.quota = int(C.K_SERVED if quota is None else quota)
        # q = 计划装/卸量（正=装车，负=卸车）。缺省与预测一致（确定性情形）。
        self.q = np.asarray(q if q is not None else (self.surplus - self.deficit), dtype=float)
        # 结算用的真实缺口（E6 里由真实需求给出；缺省等于预测缺口）
        self.truth_deficit = np.asarray(
            truth_deficit if truth_deficit is not None else self.deficit, dtype=float)
        # 解 = 「除 depot 外全部站点」的排列；depot 只作为每辆车的出发/回收点
        self.items = [i for i in range(self.n) if i != self.depot]
        self.n_items = len(self.items)

    def random_perm(self, rng):
        return rng.permutation(self.items).tolist()

    def greedy_perm(self):
        """最近邻构造解（全部非 depot 站点）。三种元启发式都用它做**暖启动**，
        保证算法结果不劣于贪心基线，比较的才是「改进幅度」而非「起跑线」。"""
        unvisited = set(self.items)
        perm, cur = [], self.depot
        while unvisited:
            nxt = min(unvisited, key=lambda j: self.sub[cur, j])
            perm.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        return perm

    def evaluate(self, perm) -> dict:
        """【唯一评估函数】三算法必须都调它，否则成本口径不可比。

        关键：**装/卸的数量由 q（按预测制定的计划量）决定**，而不是按真实缺口决定。
        这样预测误差才会真正传递到调度结果：预测偏低 → 卸货不足 → 真实缺口补不上；
        预测偏高 → 浪费车容量 → 别的站点补不上。E6 就是靠这个机制成立的。
        """
        perm = list(perm)
        travel = 0.0
        delivered = np.zeros(self.n)
        bounds = np.linspace(0, len(perm), self.fleet + 1).astype(int)
        served = 0

        for k in range(self.fleet):
            chunk = perm[bounds[k]:bounds[k + 1]]
            if not chunk:
                continue
            load = 0.0
            travel += self.sub[self.depot, chunk[0]]
            for j, s in enumerate(chunk):
                if j > 0:
                    travel += self.sub[chunk[j - 1], s]
                if served >= self.quota:                     # 出车配额用完，剩下站点本次不服务
                    continue
                served += 1
                if self.q[s] > 0:                            # 计划在此装车
                    load += min(self.q[s], self.cap - load)
                elif self.q[s] < 0:                          # 计划在此卸车
                    give = min(-self.q[s], load)
                    load -= give
                    delivered[s] += give
            travel += self.sub[chunk[-1], self.depot]

        # 服务缺口 = |真实缺口 − 实际送达| 之和：既罚「没送到」，也罚「送多了浪费车容量」。
        # 确定性情形下 delivered ≤ q = 缺口，所以 |·| 退化为 max(0, ·)，与线下模型完全一致。
        unmet = float(np.abs(self.truth_deficit - delivered).sum())
        return {"cost": travel + self.lam * unmet,
                "travel": travel, "unmet": unmet, "perm": tuple(perm)}

    def with_truth(self, truth_deficit) -> "Ctx":
        """复制一份实例，只替换「真实缺口」——用于 E6：按预测制定计划，按真值结算。"""
        return Ctx(self.sub, self.depot, self.surplus, self.deficit,
                   self.fleet, self.cap, self.lam,
                   q=self.q, truth_deficit=truth_deficit, quota=self.quota)


def build_context(pred_df: pd.DataFrame, stations: pd.DataFrame, cost: np.ndarray,
                  n_visit: int = C.N_VISIT, seed: int = 0, select_ids=None,
                  quiet: bool = False) -> Ctx:
    """从 predictions.csv 构造实例：库存取站点基础规模，需求取测试段预测均值。

    兼容两种输入：带 y_true/y_pred_rf 的 predictions.csv，或带 demand 的原始表。
    select_ids: 指定参与调度的站点（E6 需要在同一站点集合上比较不同噪声水平）。
    quiet: 批量实验时关闭打印。
    """
    truth_col = "y_true" if "y_true" in pred_df.columns else "demand"
    pred_col = "y_pred_rf" if "y_pred_rf" in pred_df.columns else truth_col

    rng = np.random.default_rng(seed)
    need = pred_df.groupby("station_id", observed=True)[truth_col].mean()
    pred = pred_df.groupby("station_id", observed=True)[pred_col].mean()
    base = stations.set_index("station_id")["base_demand"]

    ids = base.index.to_numpy()
    inv = np.round(base.reindex(ids).to_numpy() * rng.uniform(0.6, 1.4, len(ids)))
    target = np.round(pred.reindex(ids).fillna(need.reindex(ids)).to_numpy())
    diff = target - inv                                   # >0 缺车，<0 富余

    if select_ids is not None:
        sel = np.asarray(select_ids)
        pos = np.searchsorted(ids, sel)                   # ids 是排好序的站点编号
        if not np.array_equal(ids[pos], sel):
            pos = np.array([int(np.where(ids == s)[0][0]) for s in sel])
    else:
        pos = np.argsort(-np.abs(diff))[:n_visit]         # 只调度矛盾最大的站点
        sel = ids[pos]
    sub = cost[np.ix_(sel, sel)]

    d = diff[pos]
    surplus = np.clip(-d, 0, None)
    deficit = np.clip(d, 0, None)
    q = inv[pos] - target[pos]                            # 正=计划装车，负=计划卸车
    depot_pos = int(np.argmax(surplus + 1e-9))

    ctx = Ctx(sub, depot_pos, surplus, deficit, C.FLEET, C.VEH_CAP, C.LAMBDA_PENALTY,
              q=q, truth_deficit=deficit)
    if not quiet:
        print(f"[ctx] 调度 {len(sel)} 个站点，需求缺口合计 {deficit.sum():.0f} 车次，"
              f"富余合计 {surplus.sum():.0f} 车次，depot=站点{sel[depot_pos]}")
    return ctx


def truth_deficits(pred_df: pd.DataFrame, stations: pd.DataFrame, sel_ids,
                   seed: int = 0) -> np.ndarray:
    """按**真实需求**算出的缺口（E6 结算用），库存与 build_context 完全一致。"""
    truth_col = "y_true" if "y_true" in pred_df.columns else "demand"
    rng = np.random.default_rng(seed)
    target = pred_df.groupby("station_id", observed=True)[truth_col].mean()
    base = stations.set_index("station_id")["base_demand"]
    ids = base.index.to_numpy()
    inv = np.round(base.reindex(ids).to_numpy() * rng.uniform(0.6, 1.4, len(ids)))
    sel = np.asarray(sel_ids)
    pos = np.array([int(np.where(ids == s)[0][0]) for s in sel])
    t = np.round(target.reindex(sel).to_numpy())
    return np.clip(t - inv[pos], 0, None)


# --------------------------------------------------------------------------
# 基线：最近邻
# --------------------------------------------------------------------------
def greedy(ctx: Ctx):
    sol = ctx.evaluate(ctx.greedy_perm())
    return sol, [sol["cost"]]


# --------------------------------------------------------------------------
# GA
# --------------------------------------------------------------------------
def _ox_crossover(p1, p2, rng, px):
    """OX 顺序交叉：保证子代仍是合法排列（课件 W5a 的 illegal offspring 问题）。"""
    if rng.random() > px:
        return list(p1), list(p2)
    n = len(p1)
    a, b = sorted(rng.choice(n, size=2, replace=False))
    def child(p_a, p_b):
        mid = list(p_a[a:b + 1])
        rest = [x for x in p_b if x not in mid]
        return rest[:a] + mid + rest[a:]
    return child(p1, p2), child(p2, p1)


def _mutate(perm, rng, pm):
    perm = list(perm)
    if rng.random() < pm:
        i, j = rng.choice(len(perm), size=2, replace=False)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def ga(ctx: Ctx, budget: int = C.OPT_BUDGET, seed: int = 0, trace: bool = True):
    """遗传算法。种群用**最近邻解暖启动**（1 个精英）+ 随机个体，保证不劣于基线。"""
    rng = np.random.default_rng(seed)
    seed_perm = ctx.greedy_perm()
    pop = [list(seed_perm)]
    for _ in range(C.GA_POP - 1):
        p = ctx.random_perm(rng)
        if rng.random() < 0.5:                      # 一半个体是最近邻解的扰动
            p = _mutate(list(seed_perm), np.random.default_rng(rng.integers(1 << 30)), 1.0)
        pop.append(p)
    scored = sorted(((ctx.evaluate(p)["cost"], p) for p in pop), key=lambda t: t[0])
    evals, best_cost, best_perm = len(pop), scored[0][0], scored[0][1]
    hist = [best_cost]

    while evals < budget:
        # 锦标赛选择
        new_pop = [list(best_perm)]                      # 精英保留
        while len(new_pop) < C.GA_POP:
            cand = rng.integers(0, C.GA_POP, C.GA_TOURNAMENT)
            p1 = min((scored[i][1] for i in cand), key=lambda p: ctx.evaluate(p)["cost"])
            evals += C.GA_TOURNAMENT
            cand = rng.integers(0, C.GA_POP, C.GA_TOURNAMENT)
            p2 = min((scored[i][1] for i in cand), key=lambda p: ctx.evaluate(p)["cost"])
            evals += C.GA_TOURNAMENT
            c1, c2 = _ox_crossover(p1, p2, rng, C.GA_PX)
            for c in (c1, c2):
                new_pop.append(_mutate(c, rng, C.GA_PM))
                if len(new_pop) >= C.GA_POP:
                    break
            if evals >= budget:
                break

        scored = sorted(((ctx.evaluate(p)["cost"], p) for p in new_pop), key=lambda t: t[0])
        evals += len(new_pop)
        if scored[0][0] < best_cost:
            best_cost, best_perm = scored[0][0], scored[0][1]
        hist.append(best_cost)

    sol = ctx.evaluate(best_perm)
    return sol, (hist if trace else [best_cost])


# --------------------------------------------------------------------------
# SA
# --------------------------------------------------------------------------
def sa(ctx: Ctx, budget: int = C.OPT_BUDGET, seed: int = 0, trace: bool = True):
    """模拟退火：Metropolis 准则 P = exp(-(C(x')-C(x))/T)，C 越小越好。从最近邻解出发。"""
    rng = np.random.default_rng(seed)
    cur = ctx.greedy_perm()
    cur_cost = ctx.evaluate(cur)["cost"]
    best, best_cost = list(cur), cur_cost
    evals, hist = 1, [cur_cost]

    T = C.SA_T0
    while evals < budget and T > C.SA_TF:
        cand = list(cur)
        if rng.random() < 0.5:
            i, j = rng.choice(ctx.n_items, size=2, replace=False)
            cand[i], cand[j] = cand[j], cand[i]
        else:
            i, j = rng.choice(ctx.n_items, size=2, replace=False)
            cand.insert(j, cand.pop(i))
        c_cost = ctx.evaluate(cand)["cost"]
        evals += 1
        d = c_cost - cur_cost
        if d < 0 or rng.random() < math.exp(-d / max(T, 1e-9)):
            cur, cur_cost = cand, c_cost
            if cur_cost < best_cost:
                best, best_cost = list(cur), cur_cost
        hist.append(best_cost)
        T *= C.SA_COOL
        if len(hist) > budget:
            break

    return ctx.evaluate(best), (hist if trace else [best_cost])


# --------------------------------------------------------------------------
# Tabu Search
# --------------------------------------------------------------------------
def tabu(ctx: Ctx, budget: int = C.OPT_BUDGET, seed: int = 0, trace: bool = True):
    """邻域搜索 + 禁忌表。禁忌长度 L = a×Nc + Random(g)（课件公式）。从最近邻解出发。"""
    rng = np.random.default_rng(seed)
    cur = ctx.greedy_perm()
    cur_sol = ctx.evaluate(cur)
    best, best_cost = list(cur), cur_sol["cost"]
    evals, hist = 1, [best_cost]
    tabu_until = {}

    while evals < budget:
        best_nb, best_nb_cost, best_move = None, math.inf, None
        pairs = [(i, j) for i in range(ctx.n_items) for j in range(i + 1, ctx.n_items)]
        rng.shuffle(pairs)
        for (i, j) in pairs:
            if evals >= budget:
                break
            cand = list(cur)
            cand[i], cand[j] = cand[j], cand[i]
            sol = ctx.evaluate(cand)
            evals += 1
            move = (min(cand[i], cand[j]), max(cand[i], cand[j]))
            is_tabu = tabu_until.get(move, 0) > evals
            if is_tabu and sol["cost"] >= best_cost:
                continue
            if sol["cost"] < best_nb_cost:
                best_nb, best_nb_cost, best_move = sol, sol["cost"], move
        if best_nb is None:
            break
        cur, cur_sol = list(best_nb["perm"]), best_nb
        if cur_sol["cost"] < best_cost:
            best, best_cost = list(cur), cur_sol["cost"]
        nc = max(cur_sol["unmet"], 1.0)
        tenure = int(C.TABU_A * nc + rng.random() * C.TABU_G)
        tabu_until[best_move] = evals + max(tenure, 1)
        hist.append(best_cost)

    return ctx.evaluate(best), (hist if trace else [best_cost])


ALGOS = {"greedy": greedy, "ga": ga, "sa": sa, "tabu": tabu}


# --------------------------------------------------------------------------
# E5：三算法公平对比
# --------------------------------------------------------------------------
def run_e5(ctx: Ctx, seeds=None, runs: int = 3, budget: int = C.OPT_BUDGET, quiet=False):
    seeds = seeds or C.SEEDS
    rows, traces = [], {}
    for name in ("greedy", "ga", "sa", "tabu"):
        for s in seeds:
            for r in range(runs if name != "greedy" else 1):
                t0 = time.time()
                sol, hist = ALGOS[name](ctx, budget=budget, seed=s + r * 100) \
                    if name != "greedy" else greedy(ctx)
                dt = time.time() - t0
                rows.append({"algo": name, "seed": s, "run": r,
                             "final_cost": round(sol["cost"], 3),
                             "travel": round(sol["travel"], 3),
                             "unmet": round(sol["unmet"], 2),
                             "evals": budget if name != "greedy" else 1,
                             "time": round(dt, 3)})
                traces.setdefault(name, []).append(hist)
                if not quiet:
                    print(f"[E5] {name:6s} seed={s}.{r}  cost={sol['cost']:.2f}  "
                          f"travel={sol['travel']:.1f}  unmet={sol['unmet']:.0f}  {dt:.2f}s")
        if name == "greedy" and not quiet:
            print(f"[E5] greedy 仅 1 次（确定性基线）")
    out = pd.DataFrame(rows)
    out.to_csv(C.RES_DIR / "E5_compare.csv", index=False)
    return out, traces


def summarise_e5(df: pd.DataFrame) -> pd.DataFrame:
    """均值±标准差，供 Tab.3 使用。"""
    g = df.groupby("algo", observed=True).agg(
        cost_mean=("final_cost", "mean"), cost_std=("final_cost", "std"),
        travel_mean=("travel", "mean"), unmet_mean=("unmet", "mean"),
        time_mean=("time", "mean")).round(3).reset_index()
    g.to_csv(C.TAB_DIR / "tab3_optimize.csv", index=False)
    print("\n[Tab.3] 优化算法对比:\n", g.to_string(index=False))
    return g


# --------------------------------------------------------------------------
# E7：可扩展性
# --------------------------------------------------------------------------
def run_e7(sizes=(10, 30, 50), budget: int = 1000, seed: int = 0):
    """可扩展性：只构造数据与代价矩阵，**不落盘**（避免覆盖主数据集）。"""
    import data_pipeline
    import routing

    rows = []
    for n in sizes:
        df = data_pipeline.build_clean(n_stations=n, days=20, seed=seed)
        stations = data_pipeline.station_table(df)
        t0 = time.time()
        cost, _, _, _ = routing.build_cost_matrix(stations, seed=7, save=False)
        t_astar = time.time() - t0

        pred = df.copy()
        pred["y_pred_rf"] = pred["demand"] * (
            1 + np.random.default_rng(seed).normal(0, .08, len(pred)))
        ctx = build_context(pred, stations, cost,
                            n_visit=min(C.N_VISIT, n - 1), seed=seed)
        t0 = time.time()
        ga(ctx, budget=budget, seed=seed, trace=False)
        t_opt = time.time() - t0
        rows.append({"n_stations": n, "astar_time": round(t_astar, 2),
                     "opt_time": round(t_opt, 2)})
        print(f"[E7] n={n:3d}  A* {t_astar:.2f}s  GA {t_opt:.2f}s")
    out = pd.DataFrame(rows)
    out.to_csv(C.RES_DIR / "E7_scale.csv", index=False)
    return out


# --------------------------------------------------------------------------
# E6：耦合敏感性（预测误差 → 调度成本）
# --------------------------------------------------------------------------
def run_e6(pred_df: pd.DataFrame, stations: pd.DataFrame, cost: np.ndarray,
           levels=(0.0, 0.05, 0.10, 0.20), seed: int = 0, budget: int = C.OPT_BUDGET,
           rounds: int = 8):
    """耦合敏感性：**在带噪预测上做决策，在真实需求上评估该决策的代价**。

    单轮实验的敏感性很弱（一次调度的缺口变化有限），所以每档噪声跑 `rounds` 轮
    不同的库存实现并取均值±标准差，曲线才有统计意义。
    """
    truth_col = "y_true" if "y_true" in pred_df.columns else "demand"
    sel_ids = _selected_ids(pred_df, stations, seed)
    truth_df = pred_df.assign(y_pred_rf=pred_df[truth_col])

    rows = []
    for lv in levels:
        costs, regrets = [], []
        for r in range(rounds):
            s = seed * 1000 + r                      # 同一轮内 noisy 与 true 共享库存
            # ① 参照组（oracle）：用「完美预测」制定的计划，作为该轮的最优基准
            ctx_or = build_context(truth_df, stations, cost, seed=s,
                                   select_ids=sel_ids, quiet=True)
            sol_or, _ = ga(ctx_or, budget=budget, seed=s, trace=False)
            oracle_cost = ctx_or.evaluate(sol_or["perm"])["cost"]

            # ② 实验组：用「带噪预测」制定计划
            p = pred_df.copy()
            if lv > 0:
                rng = np.random.default_rng(s + 7717)
                p["y_pred_rf"] = p["y_pred_rf"] * (1 + rng.normal(0, lv, len(p)))
            ctx_plan = build_context(p, stations, cost, seed=s,
                                     select_ids=sel_ids, quiet=True)
            sol, _ = ga(ctx_plan, budget=budget, seed=s, trace=False)

            # ③ 用「真实需求」结算这份计划
            td = truth_deficits(pred_df, stations, sel_ids, seed=s)
            cost_true = ctx_plan.with_truth(td).evaluate(sol["perm"])["cost"]

            costs.append(cost_true)
            regrets.append((cost_true / oracle_cost - 1) * 100 if oracle_cost else 0.0)

        rows.append({"noise_level": round(lv, 3),
                     "final_cost": round(float(np.mean(costs)), 3),
                     "cost_std": round(float(np.std(costs)), 3),
                     "regret_pct": round(float(np.mean(regrets)), 2),
                     "regret_std": round(float(np.std(regrets)), 2),
                     "rounds": rounds})
        print(f"[E6] 噪声 {lv:4.0%} -> 真实代价 {np.mean(costs):7.2f}  "
              f"相对完美预测的损失 {np.mean(regrets):+5.1f}% ± {np.std(regrets):.1f}")
    out = pd.DataFrame(rows)
    out.to_csv(C.RES_DIR / "E6_sensitivity.csv", index=False)
    return out


def _selected_ids(pred_df: pd.DataFrame, stations: pd.DataFrame, seed: int = 0,
                  n_visit: int = C.N_VISIT):
    """复现 build_context 的站点选择，供 E6 固定候选集合。"""
    truth_col = "y_true" if "y_true" in pred_df.columns else "demand"
    pred_col = "y_pred_rf" if "y_pred_rf" in pred_df.columns else truth_col
    rng = np.random.default_rng(seed)
    pred = pred_df.groupby("station_id", observed=True)[pred_col].mean()
    base = stations.set_index("station_id")["base_demand"]
    ids = base.index.to_numpy()
    inv = np.round(base.reindex(ids).to_numpy() * rng.uniform(0.6, 1.4, len(ids)))
    target = np.round(pred.reindex(ids).fillna(base.reindex(ids)).to_numpy())
    diff = target - inv
    return ids[np.argsort(-np.abs(diff))[:n_visit]]


if __name__ == "__main__":
    import data_pipeline
    import routing

    C.ensure_dirs()
    df, stations, _ = data_pipeline.run(n_stations=20, days=15, seed=0)
    cost, stats, _, _ = routing.build_cost_matrix(stations, save=False)
    pred = df.copy()
    pred["y_pred_rf"] = pred["demand"] * (1 + np.random.default_rng(0).normal(0, .1, len(pred)))
    ctx = build_context(pred, stations, cost, n_visit=10)

    # 自测 1：三个算法都返回合法解（排列合法、成本有限）
    for name, fn in ALGOS.items():
        sol, _ = fn(ctx, budget=600, seed=0) if name != "greedy" else fn(ctx)
        assert sorted(sol["perm"]) == ctx.items, f"{name} 返回非法排列"
        assert np.isfinite(sol["cost"]), f"{name} 成本非有限"
        assert sol["unmet"] >= 0, f"{name} unmet 为负"
        print(f"[自测] {name:6s} cost={sol['cost']:.2f} travel={sol['travel']:.1f} unmet={sol['unmet']:.0f}")

    # 自测 2：GA 不应劣于随机排列（说明确实在优化）
    rng = np.random.default_rng(123)
    rand_costs = [ctx.evaluate(rng.permutation(ctx.items).tolist())["cost"] for _ in range(200)]
    ga_sol, _ = ga(ctx, budget=1500, seed=0)
    print(f"[自测] 随机解均值 {np.mean(rand_costs):.2f} vs GA {ga_sol['cost']:.2f}")
    assert ga_sol["cost"] < np.mean(rand_costs), "GA 未优于随机解"

    # 自测 3：同一 seed 可复现
    a, _ = ga(ctx, budget=800, seed=5, trace=False)
    b, _ = ga(ctx, budget=800, seed=5, trace=False)
    assert abs(a["cost"] - b["cost"]) < 1e-9, "同 seed 结果不可复现"
    print("[自测] 可复现性 OK")
    print("[自测] PASSED")
