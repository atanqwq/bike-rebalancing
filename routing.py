# -*- coding: utf-8 -*-
"""routing.py —— A* 搜索与站点代价矩阵（M4 负责，对应课程 W2–3 搜索）

为什么这里必须用 A*
-------------------
L2 优化层需要站点两两之间的**实际行驶代价**。若直接拿欧氏距离当代价，
“搜索”这一模块就变成了摆设；本文件在带障碍的网格路网上用 A* 求最短路，
得到的代价矩阵会**系统性地大于欧氏距离**（绕行系数见 results/R1_astar_stats.csv），
这正是论文 Methodology 里"为什么需要搜索"的实证依据。

输出
----
data/cost_matrix.npy        N×N float 对称矩阵
data/grid_meta.npz          网格尺寸、障碍掩码（供绘图）
results/R1_astar_stats.csv  A* 绕行系数与扩展节点数（写进 3.4 节）
"""
from __future__ import annotations

import heapq
import math

import numpy as np

import config as C

# 四邻域移动，单位步长
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


# --------------------------------------------------------------------------
# 站点布局
# --------------------------------------------------------------------------
def grid_shape(n: int) -> tuple[int, int]:
    """按站点数确定网格行列：网格单元数约为站点数的 3 倍，
    保证有足够“空路”让 A* 的绕行有意义（站点密度约 33%）。"""
    cells = max(3 * n, n + 8)
    cols = int(math.ceil(math.sqrt(cells * 1.2)))
    rows = int(math.ceil(cells / cols))
    return rows, cols


def station_coords(n: int, seed: int = 7):
    """把 n 个站点铺到网格单元中心，返回 (xs, ys)。

    坐标严格取自网格单元中心，与 A* 使用的整数单元一致，
    这样才能保证 A* 的网格代价 >= 欧氏直线距离（几何自洽）。
    """
    cells, _ = station_cells(n, seed=seed)
    xs = np.array([c + 0.5 for _, c in cells], dtype=float)
    ys = np.array([r + 0.5 for r, _ in cells], dtype=float)
    return xs, ys


def station_cells(n: int, seed: int = 7):
    """站点所在的整数网格单元（A* 的起点/终点）。"""
    rng = np.random.default_rng(seed)
    rows, cols = grid_shape(n)
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)
    if n > len(cells):
        raise ValueError(f"网格 {rows}x{cols} 容不下 {n} 个站点")
    return cells[:n], (rows, cols)


# --------------------------------------------------------------------------
# 网格路网（含随机障碍）
# --------------------------------------------------------------------------
def build_grid(n_stations: int, seed: int = 7, obstacle_p: float = 0.12):
    """构造带障碍的网格，并保证所有站点单元互相连通。"""
    cells, (rows, cols) = station_cells(n_stations, seed=seed)
    rng = np.random.default_rng(seed + 1)
    occupied = set(cells)

    blocked = np.zeros((rows, cols), dtype=bool)
    for _ in range(5):                       # 障碍过多导致不连通时逐轮减半
        blocked[:] = False
        for r in range(rows):
            for c in range(cols):
                if (r, c) in occupied:
                    continue
                if rng.random() < obstacle_p:
                    blocked[r, c] = True
        if _all_connected(blocked, cells):
            break
        obstacle_p *= 0.5
    print(f"[grid] {rows}x{cols}，障碍 {int(blocked.sum())} 格，"
          f"连通性 {'OK' if _all_connected(blocked, cells) else 'FAIL'}")
    return blocked, cells, (rows, cols)


def _bfs_reachable(blocked: np.ndarray, start):
    rows, cols = blocked.shape
    seen = np.zeros_like(blocked)
    stack = [start]
    seen[start] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not blocked[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return seen


def _all_connected(blocked: np.ndarray, cells) -> bool:
    seen = _bfs_reachable(blocked, cells[0])
    return all(seen[c] for c in cells)


# --------------------------------------------------------------------------
# A*
# --------------------------------------------------------------------------
def astar(blocked: np.ndarray, start, goal, return_expansions: bool = True):
    """四邻域 A*，启发式用曼哈顿距离（可采纳且一致）。返回 (path, cost, expansions)。"""
    rows, cols = blocked.shape

    def h(node):
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    open_heap = [(h(start), 0.0, start)]
    best_g = {start: 0.0}
    parent = {start: None}
    closed = set()
    expansions = 0

    while open_heap:
        _, g, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        closed.add(node)
        expansions += 1
        if node == goal:
            path, cur = [], node
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1], g, expansions

        r, c = node
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if blocked[nr, nc] or (nr, nc) in closed:
                continue
            ng = g + 1.0
            nxt = (nr, nc)
            if ng < best_g.get(nxt, math.inf):
                best_g[nxt] = ng
                parent[nxt] = node
                heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))

    return None, math.inf, expansions


def build_cost_matrix(stations, seed: int = 7, obstacle_p: float = 0.12, save: bool = True):
    """站点两两代价矩阵。返回 (matrix, stats 字典, blocked, cells)。"""
    C.ensure_dirs()
    n = len(stations)
    blocked, cells, shape = build_grid(n, seed=seed, obstacle_p=obstacle_p)

    mat = np.zeros((n, n), dtype=np.float64)
    expansions = []
    ratios = []
    euclid = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            path, cost, exp = astar(blocked, cells[i], cells[j])
            if path is None:
                raise RuntimeError(f"站点 {i}->{j} 不可达，请降低 obstacle_p")
            mat[i, j] = mat[j, i] = cost
            expansions.append(exp)
            dx = stations["x"].iloc[i] - stations["x"].iloc[j]
            dy = stations["y"].iloc[i] - stations["y"].iloc[j]
            d = math.hypot(dx, dy)
            euclid[i, j] = euclid[j, i] = d
            if d > 0:
                ratios.append(cost / d)

    mask = euclid > 0
    stats = {
        "n_stations": n,
        "pairs": int(n * (n - 1) / 2),
        "mean_expansions": float(np.mean(expansions)),
        "max_expansions": int(np.max(expansions)),
        "mean_cost_astar": float(mat[mask].mean()),
        "mean_cost_euclid": float(euclid[mask].mean()),
        "detour_ratio": float(np.mean(ratios)),
    }
    print(f"[A*] {stats['pairs']} 对站点；平均扩展 {stats['mean_expansions']:.1f} 个节点；"
          f"绕行系数（A*/欧氏）= {stats['detour_ratio']:.3f}")

    if save:
        np.save(C.DATA_DIR / "cost_matrix.npy", mat)
        np.savez(C.DATA_DIR / "grid_meta.npz", blocked=blocked,
                 cells=np.array(cells), shape=np.array(shape))
        import pandas as pd
        pd.DataFrame([stats]).to_csv(C.RES_DIR / "R1_astar_stats.csv", index=False)
        print(f"[save] cost_matrix.npy {mat.shape}；R1_astar_stats.csv")
    return mat, stats, blocked, cells


if __name__ == "__main__":
    import pandas as pd
    n = 12
    xs, ys = station_coords(n)
    st = pd.DataFrame({"station_id": range(n), "x": xs, "y": ys})
    mat, stats, blocked, cells = build_cost_matrix(st, save=False)

    # 自测 1：矩阵对称、对角为 0、非对角为正
    assert np.allclose(mat, mat.T), "代价矩阵不对称"
    assert np.all(np.diag(mat) == 0), "对角不为 0"
    off = mat[~np.eye(n, dtype=bool)]
    assert (off > 0).all(), "存在零代价的站点对"

    # 自测 2：A* 代价 >= 直线距离（绕行不可能比直线更短）
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = math.hypot(xs[i] - xs[j], ys[i] - ys[j])
            assert mat[i, j] >= d - 1e-9, f"A* 代价小于欧氏距离: {i},{j}"
    print(f"[自测] 矩阵 {mat.shape}，绕行系数 {stats['detour_ratio']:.3f}，全部断言通过")
    print("[自测] PASSED")
