#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

def _nice_plot(df, y_col, out_png, title, ylabel):
    if df.empty:
        print(f"[plot] skip {out_png} (empty data)")
        return
    # 只保留 y 有值的
    df = df.dropna(subset=[y_col])
    if df.empty:
        print(f"[plot] skip {out_png} (no '{y_col}' values)")
        return

    # turn 一定用整数
    df["turn"] = pd.to_numeric(df["turn"], errors="coerce")
    df = df.dropna(subset=["turn"])
    df["turn"] = df["turn"].astype(int)

    models = sorted(df["model"].unique())
    if len(models) == 0:
        print(f"[plot] skip {out_png} (no models)")
        return

    turns_sorted = sorted(df["turn"].unique())
    if len(turns_sorted) == 0:
        print(f"[plot] skip {out_png} (no turns)")
        return

    plt.figure(figsize=(8,5))
    for mk in models:
        g = df[df["model"] == mk]
        if g.empty: 
            continue
        # 对齐各 turn 的平均
        gg = g.groupby("turn", as_index=False)[y_col].mean()
        gg = gg.set_index("turn").reindex(turns_sorted)
        plt.plot(turns_sorted, gg[y_col].values, marker="o", label=mk)

    plt.title(title)
    plt.xlabel("Turn Index (t)")
    plt.ylabel(ylabel)
    plt.xticks(turns_sorted)  # 整数 tick
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[plot] saved {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate-csv", required=True, help="由 analyze_batch.py 产生的汇总 CSV")
    ap.add_argument("--out-prefix", required=True, help="输出图片前缀（不含扩展名）")
    args = ap.parse_args()

    if not os.path.exists(args.aggregate_csv):
        raise FileNotFoundError(args.aggregate_csv)
    df = pd.read_csv(args.aggregate_csv)

    # 全局空保护
    if df.empty or "model" not in df.columns or "turn" not in df.columns:
        print("[plot] aggregate is empty or missing required columns; nothing to plot.")
        return

    # 图1：视觉聚焦随 turn
    _nice_plot(
        df, y_col="focus_rate",
        out_png=args.out_prefix + "_focus_by_turn.png",
        title="Visual Focus Accuracy vs. Turn Index",
        ylabel="Focus Accuracy"
    )

    # 图2：文本正常答复率随 turn
    _nice_plot(
        df, y_col="answer_ok_rate",
        out_png=args.out_prefix + "_answer_ok_by_turn.png",
        title="Text Answer Quality vs. Turn Index",
        ylabel="Answer OK Rate"
    )

if __name__ == "__main__":
    main()
