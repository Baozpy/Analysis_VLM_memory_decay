#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize Week4 ablations.

Reads per-variant results under real_stats/week4/recap_k*/ :
- half_life.tsv   (columns: model, ..., t_half)
- mixed_effects.tsv (columns: xvar, term, coef, se, ...)

Outputs (to real_stats/week4/):
- week4_all_z.tsv                 (one row per metric_group × variant)
- week4_summary_table.tsv         (same rows, with z_mean/z_abs_mean/z_max_abs/sig)
- rank_summary.tsv                (aggregate ranks across metrics)
- week4_summary.md                (human-readable summary)
"""

import argparse
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.DataFrame()


def find_variants(out_dir: Path) -> List[Path]:
    """Find recap_k* subdirs that have at least one useful file."""
    subs = []
    for p in sorted(out_dir.glob("recap_k*")):
        if p.is_dir() and (
            (p / "mixed_effects.tsv").exists() or (p / "half_life.tsv").exists()
        ):
            subs.append(p)
    return subs


def half_life_value(hl_df: pd.DataFrame) -> Optional[float]:
    """
    Aggregate per-variant half-life as the median of t_half across models.
    Returns None if not enough usable rows.
    """
    if hl_df.empty or "t_half" not in hl_df.columns:
        return None
    vals = pd.to_numeric(hl_df["t_half"], errors="coerce")
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]  # exclude non-positive/NaN
    if vals.empty:
        return None
    return float(np.median(vals))


def mixed_effects_z(mx_df: pd.DataFrame) -> Optional[float]:
    """
    Extract z = coef / se from mixed_effects.tsv on xvar == 'B2' and term != 'const'.
    Returns None if missing / invalid.
    """
    if mx_df.empty:
        return None
    df = mx_df.copy()
    if "xvar" in df.columns:
        df = df[df["xvar"] == "B2"]
    if "term" in df.columns:
        df = df[df["term"] != "const"]
    need = {"coef", "se"}
    if not need.issubset(df.columns) or df.empty:
        return None
    coef = pd.to_numeric(df.iloc[0]["coef"], errors="coerce")
    se = pd.to_numeric(df.iloc[0]["se"], errors="coerce")
    if not (np.isfinite(coef) and np.isfinite(se)) or se == 0:
        return None
    return float(coef / se)


def summarize(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = find_variants(out_dir)

    rows_all = []
    rows_tbl = []

    # collect per-variant metrics
    for vdir in variants:
        variant = vdir.name  # e.g., recap_k12

        # half_life: median t_half
        hl = read_tsv(vdir / "half_life.tsv")
        z_hl = half_life_value(hl)
        if z_hl is not None:
            rows_all.append(dict(metric_group="half_life", variant=variant, z=z_hl))
            rows_tbl.append(
                dict(
                    metric_group="half_life",
                    variant=variant,
                    z_mean=z_hl,
                    z_abs_mean=abs(z_hl),
                    z_max_abs=abs(z_hl),
                    sig=pd.NA,  # do not mark significance for half-life
                )
            )

        # mixed_effects: coef/se
        mx = read_tsv(vdir / "mixed_effects.tsv")
        z_mx = mixed_effects_z(mx)
        if z_mx is not None:
            rows_all.append(dict(metric_group="mixed_effects", variant=variant, z=z_mx))
            rows_tbl.append(
                dict(
                    metric_group="mixed_effects",
                    variant=variant,
                    z_mean=z_mx,
                    z_abs_mean=abs(z_mx),
                    z_max_abs=abs(z_mx),
                    sig=(abs(z_mx) >= 1.96),
                )
            )

    # write all_z / summary_table
    all_z = pd.DataFrame(rows_all, columns=["metric_group", "variant", "z"])
    tbl = pd.DataFrame(
        rows_tbl,
        columns=["metric_group", "variant", "z_mean", "z_abs_mean", "z_max_abs", "sig"],
    )

    (out_dir / "week4_all_z.tsv").write_text(
        all_z.to_csv(sep="\t", index=False), encoding="utf-8"
    )
    print("[OK] wrote", out_dir / "week4_all_z.tsv", "rows:", len(all_z))

    (out_dir / "week4_summary_table.tsv").write_text(
        tbl.to_csv(sep="\t", index=False), encoding="utf-8"
    )
    print("[OK] wrote", out_dir / "week4_summary_table.tsv", "rows:", len(tbl))

    # ranking: use aggregated z_mean from tbl
    rank_rows = []
    for metric in ["mixed_effects", "half_life"]:
        sub = tbl[tbl["metric_group"] == metric].copy()
        if sub.empty:
            continue
        asc = True if metric == "mixed_effects" else False  # mixed_effects smaller is better
        sub = sub.sort_values("z_mean", ascending=asc).reset_index(drop=True)
        sub["rank"] = np.arange(1, len(sub) + 1)
        rank_rows.append(sub[["variant", "rank"]])

    if rank_rows:
        r = pd.concat(rank_rows, ignore_index=True)
        rank = (
            r.groupby("variant")["rank"]
            .agg(
                rank_mean="mean",
                rank_std="std",
                rank_min="min",
                rank_max="max",
                n_metrics="count",
            )
            .reset_index()
            .sort_values("rank_mean")
        )
    else:
        rank = pd.DataFrame(
            columns=["variant", "rank_mean", "rank_std", "rank_min", "rank_max", "n_metrics"]
        )

    (out_dir / "rank_summary.tsv").write_text(
        rank.to_csv(sep="\t", index=False), encoding="utf-8"
    )
    print("[OK] wrote", out_dir / "rank_summary.tsv")

    # markdown summary
    lines = ["# Week4 Ablations — Summary", "- Significance threshold: |Z| ≥ 1.96", ""]

    # half_life section
    sub_hl = all_z[all_z["metric_group"] == "half_life"].copy()
    if not sub_hl.empty:
        best = sub_hl.sort_values("z", ascending=False).iloc[0]
        lines += [
            "",
            "## half_life",
            f"- Best: **{best['variant']}**  (t½ ≈ {best['z']:.2f})",
            "- Top-5 by t½:",
        ]
        for _, r in sub_hl.sort_values("z", ascending=False).head(5).iterrows():
            lines.append(f"- {r['variant']}: t½≈{r['z']:.2f}")

    # mixed_effects section
    sub_mx = all_z[all_z["metric_group"] == "mixed_effects"].copy()
    if not sub_mx.empty:
        best = sub_mx.sort_values("z", ascending=True).iloc[0]
        sig_suffix = "" if abs(best["z"]) >= 1.96 else ", ns"
        lines += [
            "",
            "## mixed_effects",
            f"- Best: **{best['variant']}**  (Z̄ = {best['z']:.2f}{sig_suffix})",
            "- Top-5 by Z̄:",
        ]
        for _, r in sub_mx.sort_values("z", ascending=True).head(5).iterrows():
            sig_mark = " — significant" if abs(r["z"]) >= 1.96 else ""
            lines.append(f"- {r['variant']}: Z̄={r['z']:.2f}{sig_mark}")

    lines += ["", "## Overall pick"]
    if not rank.empty:
        top = rank.iloc[0]
        lines.append(f"- **{top['variant']}** (rank_mean = {float(top['rank_mean']):.2f})")
    else:
        lines.append("- _(no variants)_")

    lines += ["", "", "## Rank table (lower is better)",
              "variant\trank_mean\trank_std\trank_min\trank_max\tn_metrics"]
    for _, r in rank.iterrows():
        lines.append(
            f"{r['variant']}\t"
            f"{float(r['rank_mean']):.2f}\t"
            f"{(0.0 if pd.isna(r['rank_std']) else float(r['rank_std'])):.2f}\t"
            f"{float(r['rank_min']):.1f}\t"
            f"{float(r['rank_max']):.1f}\t"
            f"{int(r['n_metrics'])}"
        )

    (out_dir / "week4_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("[OK] wrote", out_dir / "week4_summary.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        default=str(Path("real_stats") / "week4"),
        help="Directory containing recap_k*/ subfolders (default: real_stats/week4)",
    )
    args = ap.parse_args()
    out = Path(args.out_dir).resolve()
    summarize(out)


if __name__ == "__main__":
    main()
