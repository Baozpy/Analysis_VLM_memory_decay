#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "real_stats" / "week4"

def md_list(items):
    return "\n".join([f" - {t}" for t in items])

def main():
    tsv = pd.read_csv(REAL/"week4_summary_table.tsv", sep="\t")
    # 逐指标取 Top-5
    md_lines = ["# Week4 Ablations — Summary", "- Significance threshold: |Z| ≥ 1.96", ""]
    for metric in ["half_life","mixed_effects","propagation"]:
        sub = tsv[tsv["metric_group"]==metric].copy().sort_values("z_mean", ascending=False)
        if sub.empty: 
            md_lines += [f"## {metric}", "(no data)", ""]
            continue
        top = sub.head(5)
        best = top.iloc[0]
        md_lines += [f"## {metric}",
                     f"- Best: **{best['variant']}**  (Z̄ = {best['z_mean']:.2f}, {'significant' if best['sig'] else 'ns'})",
                     "- Top-5 by Z̄:"]
        for _, r in top.iterrows():
            md_lines.append(f" - {r['variant']}: Z̄={r['z_mean']:.2f} (|Z|̄={r['z_abs_mean']:.2f}, |Z|max={r['z_max_abs']:.2f})")
        md_lines.append("")

    # 总评（按 rank_mean）
    ranks = pd.read_csv(REAL/"rank_summary.tsv", sep="\t").sort_values("rank_mean")
    md_lines += ["## Overall pick",
                 f"- **{ranks.iloc[0]['variant']}** (rank_mean = {ranks.iloc[0]['rank_mean']:.2f})",
                 ""]
    (REAL/"week4_summary.md").write_text("\n".join(md_lines)+"\n", encoding="utf-8")
    print("[OK] wrote", REAL/"week4_summary.md")

if __name__ == "__main__":
    main()
