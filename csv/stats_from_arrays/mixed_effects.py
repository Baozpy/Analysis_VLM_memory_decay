#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 Week2/Week3/Week4 导出的逐图像 raw 表（*_raw.tsv）做 GEE（或稳健回归）以获得混合效应估计。
输入目录由 --in-dir 指定，输出到 --out-dir/mixed_effects.tsv。

要求（任一 raw 表中）至少包含：
  model, sample_id, turn, ok 以及对应自变量列 xcol ∈ {delta, k, tau}
  - 若 delta 缺失或全 NaN，且存在 turn，则自动以 turn 兜底
  - tau 文件缺失会被跳过（不报错）
"""

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

RAW_FILES = {
    "delta": "stats_delta_raw.tsv",
    "k":     "stats_k_raw.tsv",
    "tau":   "stats_tau_raw.tsv",
}

# 仅为日志可读性，保持与你原来的标签一致：
# B2 -> delta, D -> k, B1 -> tau
LABELS = {
    "delta": "B2",
    "k":     "D",
    "tau":   "B1",
}

def _read_if_exists(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        return pd.read_csv(path, sep="\t")
    return None

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "model" in df.columns:
        df["model"] = df["model"].astype(str).str.lower()
    if "ok" in df.columns:
        # 统一 ok 为 {0,1}
        df["ok"] = (df["ok"].astype(float) > 0).astype(int)
    return df

def _ensure_delta(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """若 delta 缺失或全 NaN 且有 turn，则用 turn 兜底。返回 (df, used_fallback)"""
    df = df.copy()
    used = False
    if "delta" not in df.columns or df["delta"].isna().all():
        if "turn" in df.columns:
            df["delta"] = df["turn"]
            used = True
    return df, used

def fit_gee(df: pd.DataFrame, xcol: str) -> Tuple[pd.DataFrame, str]:
    """
    以 ok ~ xcol 做 GEE，cluster 以 sample_id。
    若失败，退化为加权最小二乘（WLS）。
    返回 (summary_df, method_name)
    """
    # 基本列检查
    need = {"ok", xcol, "sample_id"}
    if not need.issubset(df.columns):
        # 返回占位行，避免上游崩溃
        return (pd.DataFrame([{
            "term": "const", "coef": np.nan, "se": np.nan, "pval": np.nan
        }, {
            "term": xcol, "coef": np.nan, "se": np.nan, "pval": np.nan
        }]), "NA-missing-cols")

    # 过滤缺失
    df = df.dropna(subset=[xcol, "ok", "sample_id"]).copy()
    if df.empty or df[xcol].nunique() == 0:
        return (pd.DataFrame([{
            "term": "const", "coef": np.nan, "se": np.nan, "pval": np.nan
        }, {
            "term": xcol, "coef": np.nan, "se": np.nan, "pval": np.nan
        }]), "NA-empty-or-constant-x")

    df["const"] = 1.0
    endog = df["ok"].astype(float)
    exog  = df[["const", xcol]].astype(float)

    # 先 GEE，失败用 WLS 兜底
    try:
        fam = sm.families.Binomial()
        ind = sm.cov_struct.Exchangeable()
        model = sm.GEE(endog, exog, groups=df["sample_id"], family=fam, cov_struct=ind)
        res = model.fit()
        params  = res.params
        bse     = res.bse
        pvalues = res.pvalues
        method  = "GEE-Binomial-Exchangeable"
    except Exception:
        try:
            # cluster 大小近似为样本数：用样本量做权重（避免除零）
            w = df.groupby("sample_id")["ok"].transform("count").clip(lower=1).astype(float)
            model = sm.WLS(endog, exog, weights=w)
            res = model.fit()
            params  = res.params
            bse     = res.bse
            pvalues = res.pvalues
            method  = "WLS-Fallback"
        except Exception:
            # 再兜底：给出 NA 行
            return (pd.DataFrame([{
                "term": "const", "coef": np.nan, "se": np.nan, "pval": np.nan
            }, {
                "term": xcol, "coef": np.nan, "se": np.nan, "pval": np.nan
            }]), "NA-fit-failed")

    out = pd.DataFrame({
        "term": ["const", xcol],
        "coef": [params.get("const", np.nan), params.get(xcol, np.nan)],
        "se":   [bse.get("const", np.nan),   bse.get(xcol,   np.nan)],
        "pval": [pvalues.get("const", np.nan), pvalues.get(xcol, np.nan)],
    })
    return out, method

def run_block(df_raw: Optional[pd.DataFrame], xcol: str) -> Optional[pd.DataFrame]:
    """
    对某个自变量块（delta/k/tau）运行拟合。
    返回带列 xvar/method/term/coef/se/pval 的表；若无数据则返回 None。
    """
    if df_raw is None:
        print(f"[warn] raw file for {xcol} missing, skip.")
        return None

    df = _normalize_df(df_raw)

    # delta 兜底
    used_fallback = False
    if xcol == "delta":
        df, used_fallback = _ensure_delta(df)
        if used_fallback:
            print("[warn] 'delta' missing/all-NaN → fallback to 'turn' values")

    # 必要列检查（model/sample_id/ok/turn）
    base_need = {"model", "sample_id", "ok", "turn"}
    if not base_need.issubset(df.columns):
        print(f"[warn] missing base columns {base_need - set(df.columns)} for {xcol}, skip.")
        return None

    res, method = fit_gee(df, xcol)
    label = LABELS.get(xcol, xcol)
    if used_fallback:
        method = method + "+(delta->turn)"

    res.insert(0, "method", method)
    res.insert(0, "xvar", label)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    paths = {k: os.path.join(args.in_dir, v) for k, v in RAW_FILES.items()}
    pΔ = _read_if_exists(paths["delta"])
    pk = _read_if_exists(paths["k"])
    pτ = _read_if_exists(paths["tau"])

    tabs = []
    for xcol, df_raw in [("delta", pΔ), ("k", pk), ("tau", pτ)]:
        t = run_block(df_raw, xcol)
        if t is not None:
            tabs.append(t)

    if not tabs:
        # 仍然写一个空文件，避免后续脚本因为文件缺失而报错
        out = pd.DataFrame(columns=["xvar", "method", "term", "coef", "se", "pval"])
    else:
        out = pd.concat(tabs, ignore_index=True)

    out_path = os.path.join(args.out_dir, "mixed_effects.tsv")
    out.to_csv(out_path, sep="\t", index=False)
    print("[OK] saved", out_path)

if __name__ == "__main__":
    main()
