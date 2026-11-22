# -*- coding: utf-8 -*-
"""data_pipeline.py —— L0 数据层（M2 负责）

职责
----
1. 取得原始数据：真实数据（UCI Bike Sharing）或合成数据（无数据时自动降级，见 README）
2. 清洗：缺失值、极端值、零需求长时段
3. 特征工程：时间特征 + 滞后特征 + 天气特征
4. 归一化 / 离散化
5. E1：站点聚类（K-means + 肘部法 + 轮廓系数）

输出（接口契约，11/22 冻结）
----------------------------
data/clean.csv        datetime, station_id, hour, dow, is_weekend, season, weather,
                      temp, hum, windspeed, is_rush, lag_1, lag_24, roll_24, demand
data/stations.csv     station_id, x, y, base_demand, cluster
results/E1_cluster.csv        station_id, cluster
results/E1_k_selection.csv    k, silhouette, sse
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

PROFILE_FEATURES = ["hour", "dow", "is_weekend", "season", "weather",
                    "temp", "hum", "windspeed", "is_rush", "lag_1", "lag_24", "roll_24"]


# --------------------------------------------------------------------------
# 1. 数据获取
# --------------------------------------------------------------------------
def synthesize(n_stations: int = C.N_STATIONS, days: int = C.DAYS,
               start: str = C.START_DATE, seed: int = 0) -> pd.DataFrame:
    """生成合成的站点级小时需求数据（无真实数据时的默认路径）。

    结构刻意贴近 UCI Bike Sharing 的字段，方便后续无缝替换。
    """
    rng = np.random.default_rng(seed)
    n_hours = days * 24
    ts = pd.date_range(start, periods=n_hours, freq="h")

    hod = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()

    # 早晚双峰需求曲线（通勤特征）
    peak = 0.45 + 0.75 * np.exp(-((hod - 8) ** 2) / 6.0) + 0.95 * np.exp(-((hod - 18) ** 2) / 9.0)
    # 夜间低谷
    peak = np.where((hod >= 1) & (hod <= 5), peak * 0.12, peak)
    # 周末通勤峰消失，白天峰抬高
    weekend = (dow >= 5).astype(float)
    dow_factor = 1.0 - 0.35 * weekend * np.exp(-((hod - 8) ** 2) / 6.0) \
        + 0.25 * weekend * np.exp(-((hod - 14) ** 2) / 20.0)

    # 季节与天气
    season = np.tile(np.repeat([1, 2, 3, 4], 7), days)[:n_hours]
    season_factor = np.choose(np.clip(season - 1, 0, 3), [0.80, 0.95, 1.10, 0.90])
    weather = rng.choice([1, 2, 3, 4], size=n_hours, p=[0.55, 0.28, 0.12, 0.05])
    weather_factor = np.choose(weather - 1, [1.00, 0.88, 0.70, 0.45])

    temp = 18 + 9 * np.sin(2 * np.pi * (np.arange(n_hours) / (365 * 24)) - 1.3) \
        + 3 * np.sin(2 * np.pi * (hod - 9) / 24) + rng.normal(0, 1.5, n_hours)
    hum = np.clip(62 - 0.9 * (temp - 20) + rng.normal(0, 8, n_hours), 25, 98)
    windspeed = np.clip(rng.gamma(2.0, 4.0, n_hours), 0, 45)

    # 每个站点一个基础规模 + 早晚峰的轻微错位（体现站点异质性，供聚类使用）
    base = rng.lognormal(mean=np.log(4.0), sigma=0.55, size=n_stations)
    shift = rng.normal(0, 0.9, n_stations)          # 峰值时点偏移
    amp = rng.uniform(0.7, 1.4, n_stations)         # 峰值强度

    hh = hod[None, :]
    station_peak = (0.45
                    + amp[:, None] * 0.75 * np.exp(-((hh - (8 + shift[:, None])) ** 2) / 6.0)
                    + amp[:, None] * 0.95 * np.exp(-((hh - (18 + shift[:, None])) ** 2) / 9.0))
    common = peak[None, :] * dow_factor[None, :] * season_factor[None, :] * weather_factor[None, :]
    mu = np.clip(base[:, None] * station_peak * common, 0.02, None)
    demand = rng.poisson(mu * 3.0).astype(float)

    df = pd.DataFrame({
        "datetime": np.tile(ts.to_numpy(), n_stations),
        "station_id": np.repeat(np.arange(n_stations), n_hours),
        "demand": demand.reshape(-1),
    })
    df["season"] = np.tile(season, n_stations)
    df["weather"] = np.tile(weather, n_stations)
    df["temp"] = np.round(np.tile(temp, n_stations), 1)
    df["hum"] = np.round(np.tile(hum, n_stations), 1)
    df["windspeed"] = np.round(np.tile(windspeed, n_stations), 1)
    return df


def load_uci_hour(raw_path, n_stations: int = C.N_STATIONS, seed: int = 0) -> pd.DataFrame:
    """真实数据适配器：把 UCI hour.csv 摊分到 n_stations 个站点上。

    注意（论文 4.1 节必须如实说明）
    ------------------------------
    UCI Hour 数据集是**全市聚合**的逐小时计数，没有站点维度。这里按各站点权重
    做多项式摊分，属于“合成站点化”，只用于验证系统链路；若组内能取得
    城市级站点数据（如 Citi Bike trip data），应替换本函数的数据源。
    """
    raw = pd.read_csv(raw_path, parse_dates=["dteday"])
    raw["datetime"] = raw["dteday"] + pd.to_timedelta(raw["hr"], unit="h")
    raw = raw.rename(columns={"cnt": "demand"})
    raw["season"] = raw["season"].astype(int)
    raw["weather"] = raw["weathersit"].astype(int).clip(1, 4)
    raw["hum"] = raw["hum"] * 100
    raw["temp"] = raw["temp"] * 47 - 8
    raw["windspeed"] = raw["windspeed"] * 67
    base = raw[["datetime", "season", "weather", "temp", "hum", "windspeed", "demand"]].copy()

    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.full(n_stations, 3.0))
    parts = []
    for sid in range(n_stations):
        tmp = base.copy()
        tmp["station_id"] = sid
        tmp["demand"] = rng.binomial(base["demand"].to_numpy().astype(int), w[sid]).astype(float)
        parts.append(tmp)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["station_id", "datetime"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. 清洗与特征
# --------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["station_id", "datetime"]).reset_index(drop=True)

    n0 = len(df)
    df = df.dropna(subset=["demand", "datetime"])
    # 负值/异常极值截断（用 0.1%/99.9% 分位）
    lo, hi = df["demand"].quantile([0.001, 0.999])
    df["demand"] = df["demand"].clip(lo, hi)
    df["demand"] = df["demand"].round(1)
    print(f"[clean] 行数 {n0} -> {len(df)}；demand 截断区间 [{lo:.1f}, {hi:.1f}]")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["datetime"]
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_rush"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("station_id", observed=True)["demand"]
    for lag in C.LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    df["roll_24"] = g.shift(1).rolling(24, min_periods=6).mean()
    # 首尾缺失用站点均值回填，保证 predictions.csv 与 clean.csv 行数对齐
    for col in [f"lag_{k}" for k in C.LAGS] + ["roll_24"]:
        df[col] = df.groupby("station_id", observed=True)[col].transform(
            lambda s: s.fillna(s.mean()))
    return df.dropna(subset=[f"lag_{k}" for k in C.LAGS]).reset_index(drop=True)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """min-max 归一化（供 FFBPN 使用；树模型不受影响）。"""
    df = df.copy()
    for col in ["temp", "hum", "windspeed", "lag_1", "lag_24", "roll_24"]:
        if col not in df:
            continue
        lo, hi = df[col].min(), df[col].max()
        df[col + "_n"] = 0.0 if hi <= lo else (df[col] - lo) / (hi - lo)
    return df


def discretize(df: pd.DataFrame) -> pd.DataFrame:
    """按分位数把需求离散化为 低/中/高 三档（课件 W9 的 discretization）。"""
    df = df.copy()
    df["demand_level"] = pd.qcut(df["demand"], 3, labels=["low", "mid", "high"])
    return df


def build_clean(n_stations: int = C.N_STATIONS, days: int = C.DAYS,
                seed: int = 0, raw_path=None) -> pd.DataFrame:
    C.ensure_dirs()
    if raw_path is not None and str(raw_path) and str(raw_path) != "":
        import os
        if os.path.exists(raw_path):
            print(f"[build_clean] 使用真实数据: {raw_path}")
            df = load_uci_hour(raw_path, n_stations=n_stations, seed=seed)
        else:
            print(f"[build_clean] 未找到 {raw_path}，降级为合成数据")
            df = synthesize(n_stations, days, seed=seed)
    else:
        print("[build_clean] 使用合成数据（E7 可扩展性实验也用此路径）")
        df = synthesize(n_stations, days, seed=seed)

    df = clean(df)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = normalize(df)
    df = discretize(df)
    return df


# --------------------------------------------------------------------------
# 3. E1：站点聚类
# --------------------------------------------------------------------------
def station_table(df: pd.DataFrame) -> pd.DataFrame:
    """站点表：地理坐标（规则网格 + 抖动）+ 基础需求规模。"""
    from routing import station_coords
    ids = np.sort(df["station_id"].unique())
    xs, ys = station_coords(len(ids), seed=7)
    base = df.groupby("station_id", observed=True)["demand"].mean().reindex(ids).to_numpy()
    return pd.DataFrame({"station_id": ids, "x": xs, "y": ys, "base_demand": base})


def station_profiles(df: pd.DataFrame) -> np.ndarray:
    """每个站点的 24 维小时需求曲线，作为聚类的输入特征。"""
    prof = df.groupby(["station_id", "hour"], observed=True)["demand"].mean().unstack()
    prof = prof.fillna(0.0)
    prof = prof.div(prof.mean(axis=1).replace(0, 1), axis=0)   # 按站点归一，突出形状差异
    return prof.to_numpy()


def cluster_stations(profiles: np.ndarray, k_range=range(2, 11), seed: int = 0):
    """K-means + 肘部法（SSE）+ 轮廓系数，返回最优 k 与标签。"""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rows, best = [], None
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(profiles)
        sil = silhouette_score(profiles, labels) if k > 1 else 0.0
        rows.append({"k": k, "silhouette": round(float(sil), 4),
                     "sse": round(float(km.inertia_), 4)})
        if best is None or sil > best["silhouette"]:
            best = {"k": k, "silhouette": float(sil), "labels": labels}
    ks = pd.DataFrame(rows)
    print("[E1] k 选择表:\n", ks.to_string(index=False))
    print(f"[E1] 最优 k = {best['k']}（轮廓系数 {best['silhouette']:.4f}）")
    return best["k"], best["labels"], ks


# --------------------------------------------------------------------------
# 4. 落盘
# --------------------------------------------------------------------------
def save_outputs(df: pd.DataFrame, stations: pd.DataFrame,
                 labels: np.ndarray, ks: pd.DataFrame) -> None:
    C.ensure_dirs()
    cols = ["datetime", "station_id", "hour", "dow", "is_weekend", "season", "weather",
            "temp", "hum", "windspeed", "is_rush", "lag_1", "lag_24", "roll_24",
            "demand", "demand_level"] + [c for c in df.columns if c.endswith("_n")]
    df[cols].to_csv(C.DATA_DIR / "clean.csv", index=False)
    stations = stations.copy()
    stations["cluster"] = labels
    stations.round(4).to_csv(C.DATA_DIR / "stations.csv", index=False)
    stations[["station_id", "cluster"]].to_csv(C.RES_DIR / "E1_cluster.csv", index=False)
    ks.to_csv(C.RES_DIR / "E1_k_selection.csv", index=False)
    print(f"[save] clean.csv {len(df)} 行；stations.csv {len(stations)} 个站点")


def run(n_stations: int = C.N_STATIONS, days: int = C.DAYS, seed: int = 0, raw_path=None):
    df = build_clean(n_stations=n_stations, days=days, seed=seed, raw_path=raw_path)
    stations = station_table(df)
    k, labels, ks = cluster_stations(station_profiles(df), seed=seed)
    stations["cluster"] = labels
    save_outputs(df, stations, labels, ks)
    return df, stations, ks


if __name__ == "__main__":
    d, s, k_tab = run()
    print("\n[自测] clean 列:", list(d.columns))
    print("[自测] 各档需求计数:\n", d["demand_level"].value_counts().to_string())
    print("[自测] 站点簇分布:\n", s["cluster"].value_counts().sort_index().to_string())
    assert d["demand"].isna().sum() == 0, "存在缺失值"
    assert len(s) == d["station_id"].nunique(), "站点数与数据不一致"
    print("\n[自测] PASSED")
