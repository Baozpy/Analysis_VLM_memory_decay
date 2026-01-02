#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_decay_by_image.py

功能：
1. 读取每个模型的 *_analysis.csv （我们之前导出的那种）
2. 过滤掉无效 row（比如没有 target_img 的行、多图比较题、无法判断正误的行）
3. 对同一张图片被“第1次/第2次/第3次/...提问”时模型是否 still on-focus 进行统计
   -> focus_correct == 1 视为记住了正确图像
4. 对不同图片取平均，得到该模型在第k次回访时的平均记忆保持率
5. 把不同模型的曲线画在同一张图里并保存

注意：
- 不会写回 CSV，不会覆盖 *_analysis.csv
- 输出图名包含 sample id，避免冲突
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_and_filter(csv_path):
    """
    读取并清洗单个模型的 analysis CSV。
    仅保留：
      - target_img 非空 (说明题目是在问某一张明确的图，而不是"比较Image1和Image4")
      - focus_correct 非空 (我们能判断它有没有对焦)
    其他行直接丢弃（这些行仍保留在原csv，但不进入记忆衰减分析）
    """
    df = pd.read_csv(csv_path)

    # 统一列名防御：有时可能是字符串 "None" 或 NaN
    def _coerce_numeric(col):
        if col not in df.columns:
            return None
        out = pd.to_numeric(df[col], errors="coerce")
        return out

    target_img = _coerce_numeric("target_img")
    focus_correct = _coerce_numeric("focus_correct")

    if target_img is None or focus_correct is None:
        # 如果缺关键列，直接返回空 df
        print(f"[WARN] {csv_path} 缺少 target_img 或 focus_correct，返回空")
        return pd.DataFrame(columns=["target_img", "focus_correct"])

    df["target_img_num"] = target_img
    df["focus_correct_num"] = focus_correct

    # 过滤条件：两个列都不是 NaN
    mask_valid = (~df["target_img_num"].isna()) & (~df["focus_correct_num"].isna())
    df_valid = df[mask_valid].copy()

    # 再保险：把它们转成 int/bool-ish
    df_valid["target_img_num"] = df_valid["target_img_num"].astype(int)
    df_valid["focus_correct_num"] = (df_valid["focus_correct_num"].astype(float) > 0.5).astype(int)

    return df_valid


def build_decay_curve(df_valid):
    """
    输入：清洗过的 df_valid（单个模型）
      必须至少含有:
        - target_img_num: 问的是哪张图 (1,2,3,4,...)
        - focus_correct_num: 0/1，该轮是否对焦到正确图像

    我们要做：
      对于每个 target_img_num = i
        找到所有问这张图的轮次（按出现顺序）
        记录第1次提问是否对焦、第2次提问是否对焦、... -> 序列
      然后对所有图片在相同“回访索引”上求平均

    输出：
      visits_idx: [1,2,3,...]  (第几次问同一张图)
      retention:  [x1,x2,x3,...]  (在第k次问时，模型平均还能保持正确聚焦的概率)
    """

    if df_valid.empty:
        return [], []

    # 我们默认 df_valid 当前顺序就是对话自然顺序
    # 如果 csv 里有 turn_index 之类的列，可以按它 sort
    # 尝试自动检测：
    turn_col_candidates = ["turn", "turn_index", "round", "step"]
    turn_col = None
    for c in turn_col_candidates:
        if c in df_valid.columns:
            turn_col = c
            break

    if turn_col is not None:
        df_valid = df_valid.sort_values(by=turn_col)
    else:
        # 如果没有 turn 列，那就相信原顺序
        df_valid = df_valid.copy()

    # 对每张图片分别收集focus_correct的序列
    per_image_focus = {}
    for img_id, df_img in df_valid.groupby("target_img_num"):
        # 按顺序取该图的回答是否对焦
        seq = df_img["focus_correct_num"].tolist()
        per_image_focus[img_id] = seq

    # 现在我们要对齐这些序列（不同图片可能被问的次数不同）
    max_visits = max(len(seq) for seq in per_image_focus.values())

    # 对每个访问序号k，收集所有图片的第k次问答是否正确
    retention_curve = []
    for k in range(max_visits):
        vals_at_k = []
        for img_id, seq in per_image_focus.items():
            if k < len(seq):  # 这张图至少被问到第k+1次
                vals_at_k.append(seq[k])
        if len(vals_at_k) == 0:
            retention_curve.append(np.nan)
        else:
            retention_curve.append(float(np.mean(vals_at_k)))

    visits_idx = list(range(1, max_visits + 1))
    return visits_idx, retention_curve


def plot_models(curves_dict, out_path, sample_id=None):
    """
    curves_dict: { model_name: (visits_idx, retention_curve) }
    out_path: 输出图片路径
    """

    plt.figure(figsize=(8,5), dpi=150)

    for model_name, (x, y) in curves_dict.items():
        if len(x) == 0:
            print(f"[WARN] {model_name} 没有有效数据，跳过画线")
            continue
        plt.plot(x, y, marker="o", label=model_name)

    plt.ylim(-0.05, 1.05)
    plt.xlabel("Revisit index k (k = 1 is first mention of the image)")
    plt.ylabel("Proportion still focusing on correct image (focus_correct rate)")
    title = "Visual Memory Retention"
    if sample_id is not None:
        title += f" (sample {sample_id})"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path)
    print(f"[OK] 图已保存到 {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", type=str, default="unknown",
                        help="只是用于保存图名/标题，不参与计算，比如 '8'")
    parser.add_argument("--blip2", type=str, default=None,
                        help="blip2_sampleX_analysis.csv 路径")
    parser.add_argument("--llava", type=str, default=None,
                        help="llava_sampleX_analysis.csv 路径")
    parser.add_argument("--qwen", type=str, default=None,
                        help="qwen_sampleX_analysis.csv 路径")
    parser.add_argument("--out", type=str, default=None,
                        help="输出图片名，默认 memory_decay_by_image_sample{sample-id}.png")
    args = parser.parse_args()

    curves_dict = {}

    if args.blip2:
        df_b = load_and_filter(args.blip2)
        x_b, y_b = build_decay_curve(df_b)
        curves_dict["BLIP2"] = (x_b, y_b)

    if args.llava:
        df_l = load_and_filter(args.llava)
        x_l, y_l = build_decay_curve(df_l)
        curves_dict["LLaVA"] = (x_l, y_l)

    if args.qwen:
        df_q = load_and_filter(args.qwen)
        x_q, y_q = build_decay_curve(df_q)
        curves_dict["Qwen"] = (x_q, y_q)

    # 输出文件名
    if args.out is not None:
        out_path = Path(args.out)
    else:
        out_path = Path(f"memory_decay_by_image_sample{args.sample_id}.png")

    plot_models(curves_dict, out_path, sample_id=args.sample_id)


if __name__ == "__main__":
    main()
