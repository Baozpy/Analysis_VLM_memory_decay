#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust half-life fitter.
优先读取 stats_Δ.tsv(model, delta, ok_mean)，
否则退回 stats_delta_raw.tsv -> groupby(delta)；若无 delta 列就用 turn 兜底，
再不行退回 events.tsv(model, turn, ok)。
拟合: log(ok_mean) = log(A) - lambda * delta  （对数线性回归）
输出: half_life.tsv: model, points, delta_min, delta_max, A, lambda_, t_half
"""
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd

def load_delta_ok_mean(in_dir: Path) -> pd.DataFrame:
    """返回列: model, delta, ok_mean（非空）"""
    pΔ = in_dir / "stats_Δ.tsv"
    if pΔ.exists():
        df = pd.read_csv(pΔ, sep="\t")
        need = {"model","delta","ok_mean"}
        if need.issubset(df.columns) and len(df):
            return df[["model","delta","ok_mean"]].copy()

    # fallback1: stats_delta_raw.tsv
    praw = in_dir / "stats_delta_raw.tsv"
    if praw.exists():
        df = pd.read_csv(praw, sep="\t")
        need_base = {"model","turn","ok"}
        if "delta" not in df.columns:
            # 用 turn 兜底
            if need_base.issubset(df.columns) and len(df):
                tmp = df.rename(columns={"turn":"delta"})
                g = (tmp.groupby(["model","delta"], as_index=False)["ok"]
                        .mean().rename(columns={"ok":"ok_mean"}))
                return g
        else:
            need = {"model","delta","ok"}
            if need.issubset(df.columns) and len(df):
                g = (df.groupby(["model","delta"], as_index=False)["ok"]
                       .mean().rename(columns={"ok":"ok_mean"}))
                return g

    # fallback2: events.tsv （逐回合事件）
    pevt = in_dir / "events.tsv"
    if pevt.exists():
        df = pd.read_csv(pevt, sep="\t")
        need = {"model","turn","ok"}
        if need.issubset(df.columns) and len(df):
            tmp = df.rename(columns={"turn":"delta"})
            g = (tmp.groupby(["model","delta"], as_index=False)["ok"]
                    .mean().rename(columns={"ok":"ok_mean"}))
            return g

    # 彻底失败
    return pd.DataFrame(columns=["model","delta","ok_mean"])

def fit_one(group: pd.DataFrame) -> dict:
    """对单一 model 的 (delta, ok_mean) 做 log 线性回归，返回参数与 t_half。"""
    g = group.dropna(subset=["delta","ok_mean"]).copy()
    if g.empty:
        return None
    # 只保留 delta 非负、ok_mean 在 (0,1] 内的点
    g = g[(g["delta"] >= 0)].copy()
    if g.empty:
        return None

    # clip 到 (0,1]，避免 log(0)
    eps = 1e-6
    y = np.clip(g["ok_mean"].astype(float).to_numpy(), eps, 1.0)
    x = g["delta"].astype(float).to_numpy()

    # 线性回归: log(y) = b - λ x
    Y = np.log(y)
    if x.size < 2 or np.allclose(x.var(), 0):
        # 点太少/横坐标无变化，给出退化估计
        A = float(np.exp(Y.mean()))
        lam = np.nan
        t_half = np.nan
    else:
        m, b = np.polyfit(x, Y, 1)  # Y ≈ m*x + b
        lam = float(-m)
        A = float(np.exp(b))
        t_half = (np.log(2)/lam) if (lam is not None and np.isfinite(lam) and lam>0) else np.nan

    return {
        "points": int(len(g)),
        "delta_min": float(np.nanmin(x)) if x.size else np.nan,
        "delta_max": float(np.nanmax(x)) if x.size else np.nan,
        "A": A,
        "lambda_": lam,
        "t_half": t_half,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_delta_ok_mean(in_dir)
    if df.empty:
        # 写一个空文件也行，但明确提示
        print(f"[warn] no usable delta/ok_mean in {in_dir}, half_life.tsv skipped.")
        (out_dir/"half_life.tsv").write_text("", encoding="utf-8")
        return

    rows = []
    for model, g in df.groupby("model"):
        res = fit_one(g)
        if res is None:
            continue
        res["model"] = model
        rows.append(res)

    if not rows:
        print(f"[warn] half-life fit produced no rows for {in_dir}")
        (out_dir/"half_life.tsv").write_text("", encoding="utf-8")
        return

    out = pd.DataFrame(rows, columns=["model","points","delta_min","delta_max","A","lambda_","t_half"])
    out.to_csv(out_dir/"half_life.tsv", sep="\t", index=False)
    print("[OK] saved", out_dir/"half_life.tsv")

if __name__ == "__main__":
    main()
