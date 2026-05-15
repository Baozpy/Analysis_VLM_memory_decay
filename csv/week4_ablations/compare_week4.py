#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "real_stats" / "week4"

def collect(metric_file, metric_group):
    rows = []
    for vdir in sorted(REAL.iterdir()):
        if not vdir.is_dir(): continue
        p = vdir / metric_file
        if not p.exists(): continue
        df = pd.read_csv(p, sep="\t")
        # 与 week3 的 summarize 兼容：如果没有显式 z 列，挑数值列当“z 代理”
        z_cols = [c for c in df.columns if c not in ("model","term","method","variant") and pd.api.types.is_numeric_dtype(df[c])]
        if "z" in df.columns:
            zs = df["z"]
        else:
            # half_life: 取 t_half 作为 z 的代理（越大越“强”），其余按经验取1-2个最核心列
            prefer = {"half_life":["t_half","A","lambda_"],
                      "mixed_effects":["coef"],
                      "propagation":["n_edges","recovery_rate"]}
            picks = [c for c in prefer.get(metric_group, []) if c in z_cols] or z_cols
            zs = df[picks].select_dtypes("number").sum(axis=1)
        rows.append(pd.DataFrame({"metric_group":metric_group, "variant":df["variant"], "z":zs}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["metric_group","variant","z"])

def main():
    out_z = []
    out_z.append(collect("Z_compare_half_life.tsv", "half_life"))
    out_z.append(collect("Z_compare_mixed_effects.tsv", "mixed_effects"))
    out_z.append(collect("Z_compare_propagation.tsv", "propagation"))
    zdf = pd.concat(out_z, ignore_index=True)
    zdf.to_csv(REAL/"week4_all_z.tsv", sep="\t", index=False)

    # 汇总统计
    g = zdf.groupby(["metric_group","variant"])["z"]
    summary = (pd.DataFrame({
        "z_mean": g.mean(),
        "z_abs_mean": g.apply(lambda s: s.abs().mean()),
        "z_max_abs": g.apply(lambda s: s.abs().max()),
    }).reset_index())
    summary["sig"] = (summary["z_abs_mean"] >= 1.96)  # 与 week3 一致阈值
    summary.to_csv(REAL/"week4_summary_table.tsv", sep="\t", index=False)

    # 排名表（越大越好）
    ranks = (summary.assign(rank=summary.groupby("metric_group")["z_mean"].rank(method="average", ascending=False))
                    .groupby("variant").agg(rank_mean=("rank","mean"),
                                            rank_std =("rank","std"),
                                            rank_min =("rank","min"),
                                            rank_max =("rank","max"),
                                            n_metrics=("rank","size")).reset_index())
    ranks.to_csv(REAL/"rank_summary.tsv", sep="\t", index=False)

    print("[OK] wrote", REAL/"week4_all_z.tsv")
    print("[OK] wrote", REAL/"week4_summary_table.tsv")
    print("[OK] wrote", REAL/"rank_summary.tsv")

if __name__ == "__main__":
    main()
