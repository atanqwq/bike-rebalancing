# -*- coding: utf-8 -*-
"""forecast.py —— L1 预测层（M3 负责，对应课程 W5b 回归 + W7 机器学习）

模型
----
linear / ridge / lasso   : 线性基线（W5b）
rf                       : 随机森林（W7 集成学习）
ffbpn                    : 前馈反向传播网络，用 MLP 实现（W7 FFBPN）

两个关键方法学决定（论文 3.3 节必须写清）
------------------------------------------
1. **时序数据一律用 expanding-window 的 TimeSeriesSplit**，禁止随机 KFold。
   这里先对“唯一时间戳”做 TimeSeriesSplit 得到折边界，再映射回行，
   避免同一小时的不同站点同时出现在训练集与测试集（那会造成泄漏）。
2. 超参网格只在 seed=0 上选一次，之后固定配置跑全部种子，保证公平。

输出
----
results/E2_baseline.csv    model, seed, rmse, mae, r2
results/E3_main.csv        model, seed, rmse, mae, r2, train_time, config
results/E4_ablation.csv    feature_set, rmse, mae, r2, delta_r2
results/R2_feat_importance.csv  feature, importance
data/predictions.csv       datetime, station_id, y_true, y_pred_<model>...
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import config as C

FEATURES = ["hour", "dow", "is_weekend", "season", "weather",
            "temp", "hum", "windspeed", "is_rush", "lag_1", "lag_24", "roll_24"]

FEATURE_SETS = {
    "full": FEATURES,
    "no_weather": [f for f in FEATURES if f not in
                   ("season", "weather", "temp", "hum", "windspeed")],
    "no_time": [f for f in FEATURES if f not in
                ("hour", "dow", "is_weekend", "is_rush")],
    "lag_only": ["lag_1", "lag_24", "roll_24"],
}


def load_clean() -> pd.DataFrame:
    path = C.DATA_DIR / "clean.csv"
    if not path.exists():
        import data_pipeline
        return data_pipeline.run()[0]
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df.sort_values(["datetime", "station_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 时序折（对唯一时间戳切分，再映射到行）
# --------------------------------------------------------------------------
def make_folds(df: pd.DataFrame, n_splits: int = C.CV_SPLITS):
    from sklearn.model_selection import TimeSeriesSplit
    times = np.sort(df["datetime"].unique())
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for tr_idx, te_idx in tscv.split(times):
        tr_times = pd.Index(times[tr_idx])
        te_times = pd.Index(times[te_idx])
        tr_mask = df["datetime"].isin(tr_times).to_numpy()
        te_mask = df["datetime"].isin(te_times).to_numpy()
        folds.append((tr_mask, te_mask))
    return folds


# --------------------------------------------------------------------------
# 模型工厂
# --------------------------------------------------------------------------
def build_model(name: str, seed: int = 0, **kw):
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.compose import TransformedTargetRegressor

    scaled = lambda est: Pipeline([("scaler", StandardScaler()), ("est", est)])  # noqa: E731

    if name == "linear":
        return scaled(LinearRegression())
    if name == "ridge":
        return scaled(Ridge(alpha=kw.get("alpha", 1.0), random_state=seed))
    if name == "lasso":
        return scaled(Lasso(alpha=kw.get("alpha", 0.5), max_iter=5000, random_state=seed))
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=kw.get("n_estimators", 120),
            max_depth=kw.get("max_depth", None),
            min_samples_leaf=kw.get("min_samples_leaf", 1),
            n_jobs=C.N_JOBS, random_state=seed)
    if name == "ffbpn":
        mlp = MLPRegressor(
            hidden_layer_sizes=kw.get("hidden_layer_sizes", (64, 32)),
            alpha=kw.get("alpha", 1e-4),
            activation="relu", solver="adam",
            max_iter=kw.get("max_iter", 250),
            early_stopping=True, n_iter_no_change=8,
            random_state=seed)
        return TransformedTargetRegressor(
            regressor=scaled(mlp), transformer=StandardScaler())
    raise ValueError(f"未知模型: {name}")


def _metrics(y, yhat) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {"rmse": float(np.sqrt(mean_squared_error(y, yhat))),
            "mae": float(mean_absolute_error(y, yhat)),
            "r2": float(r2_score(y, yhat))}


def evaluate_cv(df: pd.DataFrame, name: str, seed: int = 0, folds=None, **kw):
    """在时序折上评估，返回（各折平均指标, 训练耗时秒, 每折指标列表）。"""
    folds = folds if folds is not None else make_folds(df)
    Xall = df[FEATURES].to_numpy(dtype=np.float32)
    yall = df["demand"].to_numpy(dtype=np.float32)

    per_fold, fits = [], 0
    t0 = time.time()
    for tr, te in folds:
        model = build_model(name, seed=seed, **kw)
        model.fit(Xall[tr], yall[tr])
        m = _metrics(yall[te], model.predict(Xall[te]))
        per_fold.append(m)
        fits += 1
    dt = time.time() - t0
    avg = {k: float(np.mean([m[k] for m in per_fold])) for k in ("rmse", "mae", "r2")}
    avg["folds"] = fits
    return avg, dt, per_fold


# --------------------------------------------------------------------------
# 实验
# --------------------------------------------------------------------------
def run_e2_baseline(df, seeds=None, folds=None, quiet=False):
    seeds = seeds or C.SEEDS
    folds = folds if folds is not None else make_folds(df)
    rows = []
    for name in ("linear", "ridge", "lasso"):
        for s in seeds:
            m, dt, _ = evaluate_cv(df, name, seed=s, folds=folds)
            rows.append({"model": name, "seed": s, "rmse": round(m["rmse"], 4),
                         "mae": round(m["mae"], 4), "r2": round(m["r2"], 4)})
            if not quiet:
                print(f"[E2] {name:6s} seed={s}  RMSE={m['rmse']:.3f}  R2={m['r2']:.3f}")
    out = pd.DataFrame(rows)
    C.ensure_dirs()
    out.to_csv(C.RES_DIR / "E2_baseline.csv", index=False)
    return out


def select_config(df, name: str, grid, seed: int = 0, n_splits: int = 3):
    """超参选择：只在 seed=0 上做，选完即固定（保证后续对比公平）。"""
    times = np.sort(df["datetime"].unique())
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    sub_folds = []
    for tr_idx, te_idx in tscv.split(times):
        tr_mask = df["datetime"].isin(pd.Index(times[tr_idx])).to_numpy()
        te_mask = df["datetime"].isin(pd.Index(times[te_idx])).to_numpy()
        sub_folds.append((tr_mask, te_mask))

    best, best_r2 = None, -np.inf
    for cfg in grid:
        m, _, _ = evaluate_cv(df, name, seed=seed, folds=sub_folds, **cfg)
        print(f"[select] {name} {cfg} -> R2={m['r2']:.3f}")
        if m["r2"] > best_r2:
            best, best_r2 = cfg, m["r2"]
    print(f"[select] 选定 {name}: {best} (R2={best_r2:.3f})")
    return best


def run_e3_main(df, seeds=None, folds=None, fast: bool = False, quiet=False):
    seeds = seeds or C.SEEDS
    folds = folds if folds is not None else make_folds(df)
    rf_grid = C.RF_GRID[:1] if fast else C.RF_GRID
    mlp_grid = C.MLP_GRID[:1] if fast else C.MLP_GRID
    rf_cfg = select_config(df, "rf", rf_grid)
    mlp_cfg = select_config(df, "ffbpn", mlp_grid)

    rows, fitted = [], {}
    for name, cfg in (("rf", rf_cfg), ("ffbpn", mlp_cfg)):
        for s in seeds:
            m, dt, _ = evaluate_cv(df, name, seed=s, folds=folds, **cfg)
            rows.append({"model": name, "seed": s, "rmse": round(m["rmse"], 4),
                         "mae": round(m["mae"], 4), "r2": round(m["r2"], 4),
                         "train_time": round(dt, 2), "config": str(cfg)})
            if not quiet:
                print(f"[E3] {name:6s} seed={s}  RMSE={m['rmse']:.3f}  R2={m['r2']:.3f}  {dt:.1f}s")
    out = pd.DataFrame(rows)
    out.to_csv(C.RES_DIR / "E3_main.csv", index=False)
    return out, {"rf": rf_cfg, "ffbpn": mlp_cfg}


def run_e4_ablation(df, rf_cfg, seed: int = 0, folds=None):
    folds = folds if folds is not None else make_folds(df)
    Xall = df[FEATURES].to_numpy(dtype=np.float32)
    yall = df["demand"].to_numpy(dtype=np.float32)
    idx = {f: i for i, f in enumerate(FEATURES)}

    rows, base_r2 = [], None
    for set_name, feats in FEATURE_SETS.items():
        cols = [idx[f] for f in feats]
        per = []
        for tr, te in folds:
            model = build_model("rf", seed=seed, **rf_cfg)
            model.fit(Xall[tr][:, cols], yall[tr])
            per.append(_metrics(yall[te], model.predict(Xall[te][:, cols])))
        m = {k: float(np.mean([p[k] for p in per])) for k in ("rmse", "mae", "r2")}
        if base_r2 is None:
            base_r2 = m["r2"]
        rows.append({"feature_set": set_name, "rmse": round(m["rmse"], 4),
                     "mae": round(m["mae"], 4), "r2": round(m["r2"], 4),
                     "delta_r2": round(m["r2"] - base_r2, 4)})
        print(f"[E4] {set_name:11s} R2={m['r2']:.3f}  ΔR2={m['r2'] - base_r2:+.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(C.RES_DIR / "E4_ablation.csv", index=False)
    return out


def feature_importance(df, rf_cfg, seed: int = 0):
    """随机森林特征重要性（写进 4.3 节，支撑 E4 的解释）。"""
    model = build_model("rf", seed=seed, **rf_cfg)
    model.fit(df[FEATURES].to_numpy(dtype=np.float32),
              df["demand"].to_numpy(dtype=np.float32))
    out = pd.DataFrame({"feature": FEATURES,
                        "importance": np.round(model.feature_importances_, 4)})
    out = out.sort_values("importance", ascending=False).reset_index(drop=True)
    out.to_csv(C.RES_DIR / "R2_feat_importance.csv", index=False)
    print("[RF] 特征重要性 Top5:\n", out.head(5).to_string(index=False))
    return out


def predict_all(df, configs, seed: int = 0, test_frac: float = C.TEST_SIZE):
    """在最后 test_frac 的时间段上给出各模型的预测，落盘 predictions.csv。

    这是 M4、M5 的输入：E6 耦合敏感性实验会向这里的预测注入误差。
    """
    times = np.sort(df["datetime"].unique())
    cut = times[int(len(times) * (1 - test_frac))]
    tr = (df["datetime"] < cut).to_numpy()
    te = ~tr

    Xall = df[FEATURES].to_numpy(dtype=np.float32)
    yall = df["demand"].to_numpy(dtype=np.float32)
    out = df.loc[te, ["datetime", "station_id"]].copy()
    out["y_true"] = yall[te]

    for name in ("linear", "rf", "ffbpn"):
        cfg = configs.get(name, {})
        model = build_model(name, seed=seed, **cfg)
        model.fit(Xall[tr], yall[tr])
        out[f"y_pred_{name}"] = model.predict(Xall[te])
        m = _metrics(yall[te], out[f"y_pred_{name}"].to_numpy())
        print(f"[predict] {name:6s} 测试段 RMSE={m['rmse']:.3f}  R2={m['r2']:.3f}")

    out.to_csv(C.DATA_DIR / "predictions.csv", index=False)
    print(f"[save] predictions.csv {len(out)} 行（测试段占 {test_frac:.0%}）")
    return out


def run(fast: bool = False):
    df = load_clean()
    seeds = [0] if fast else C.SEEDS
    folds = make_folds(df, n_splits=2 if fast else C.CV_SPLITS)
    e2 = run_e2_baseline(df, seeds=seeds, folds=folds)
    e3, cfgs = run_e3_main(df, seeds=seeds, folds=folds, fast=fast)
    e4 = run_e4_ablation(df, cfgs["rf"], folds=folds)
    feature_importance(df, cfgs["rf"])
    predict_all(df, {**cfgs, "linear": {}}, seed=0)
    return {"e2": e2, "e3": e3, "e4": e4, "configs": cfgs}


if __name__ == "__main__":
    import sys
    fast = "--fast" in sys.argv
    res = run(fast=fast)
    tab1 = pd.concat([res["e2"], res["e3"]], ignore_index=True)
    tab1 = tab1.groupby("model", as_index=False)[["rmse", "mae", "r2"]].mean().round(4)
    tab1.to_csv(C.TAB_DIR / "tab1_forecast.csv", index=False)
    print("\n[Tab.1] 预测模型对比（各种子均值）:\n", tab1.to_string(index=False))
    best = tab1.sort_values("r2", ascending=False).iloc[0]
    print(f"\n[自测] 最优模型 = {best['model']}（R2={best['r2']}）")
    assert best["r2"] > 0.5, "所有模型 R2 均过低，检查特征或数据"
    print("[自测] PASSED")
