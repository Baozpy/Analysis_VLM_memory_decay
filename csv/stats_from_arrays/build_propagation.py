#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_propagation.py (Argparse version)

从 events.tsv 中构建“错误传播”统计：
- 逐 (model, sample_id, image) 的时间序列，追踪 OK 从 1->0 的首次跌落，
  直到恢复为 1（或序列结束），生成一条传播边。
- 输出：
  1) prop_edges.tsv：每条传播边（model, sample_id, img, src_turn, dst_turn, length, recovered）
  2) propagation_stats.tsv：按模型聚合的统计（链条数、平均长度、恢复率等）

兼容性：
- 若没有 images 列，则尝试 img / image；
- 若 images 是字符串，如 "[1,2]" 或 "1,2" 或 "1;2" 都能解析；
- 若确实没有图像信息，也能退化为“整轮对话”的传播（img 记为 -1）。
"""

import os
import re
import ast
import argparse
import pandas as pd
from typing import List, Tuple, Any, Optional


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-dir", required=True,
        help="输入目录，内含 events.tsv（由 build_stats_from_samples.py 生成）"
    )
    ap.add_argument(
        "--out-dir", required=True,
        help="输出目录，将写出 prop_edges.tsv / propagation_stats.tsv"
    )
    return ap.parse_args()


def _parse_images_cell(x: Any) -> List[int]:
    """
    尽量鲁棒地把一格“images”解析成整数列表。
    支持：
      - 已经是 list
      - 字符串："[1,2]"、"1,2"、"1;2"、"1"
      - 空或异常 -> []
    """
    if isinstance(x, list):
        # 有些情况下已经是 list[int]
        out = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                pass
        return out

    if isinstance(x, (int, float)):
        # 单个数字
        try:
            return [int(x)]
        except Exception:
            return []

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        # 尝试 AST 解析 "[1,2]"
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = ast.literal_eval(s)
                out = []
                for v in arr:
                    try:
                        out.append(int(v))
                    except Exception:
                        pass
                return out
            except Exception:
                pass
        # 尝试逗号/分号分隔
        if "," in s or ";" in s:
            parts = re.split(r"[;,]\s*", s)
            out = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                try:
                    out.append(int(p))
                except Exception:
                    pass
            return out
        # 尝试单个数字字符串
        try:
            return [int(s)]
        except Exception:
            return []

    # 其他类型 -> 空
    return []


def explode_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    把 events（每行可能对应多个 images）展开为逐 image 的行。
    需要列：model, sample_id, turn, ok, images(可选)
    """
    # 规范列名
    cols = {c.lower(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    model_c = pick("model")
    sample_c = pick("sample_id", "sid", "sample")
    turn_c = pick("turn", "t")
    ok_c = pick("ok", "answer_ok", "answer_ok_rate")  # events.tsv 中一般是 ok（0/1）
    images_c = pick("images", "image", "img")

    # 基本检查
    need = [model_c, sample_c, turn_c, ok_c]
    miss = [c for c in need if c is None]
    if miss:
        raise ValueError(f"[explode_events] 缺少关键列: {miss}")

    # 若没有 images 列，则创建一个退化列（每行唯一 img=-1）
    if images_c is None:
        df = df.copy()
        df["__images_fallback__"] = [[]] * len(df)  # 空列表 -> 稍后填 -1
        images_c = "__images_fallback__"

    rows = []
    for _, r in df.iterrows():
        model = r[model_c]
        sid = int(r[sample_c])
        turn = int(r[turn_c])
        ok = float(r[ok_c])

        imgs = r[images_c]
        imgs = _parse_images_cell(imgs)
        if len(imgs) == 0:
            imgs = [-1]  # 没图像信息时，退化为单一轨迹

        for img in imgs:
            rows.append({
                "model": model,
                "sample_id": sid,
                "img": int(img),
                "turn": turn,
                "ok": float(ok)
            })

    out = pd.DataFrame(rows)
    return out.sort_values(["model", "sample_id", "img", "turn"]).reset_index(drop=True)


def find_propagation_edges(seq: pd.DataFrame) -> List[dict]:
    """
    输入：单条轨迹（同一 model, sample_id, img）按 turn 升序的 DataFrame，含列：turn, ok
    输出：从 1->0 的错误传播边，直到恢复 1（或序列结束）
    """
    # 只保留必须列
    seq = seq[["turn", "ok"]].sort_values("turn").reset_index(drop=True)

    edges = []
    in_chain = False
    start_turn: Optional[int] = None

    for i in range(len(seq)):
        turn_i = int(seq.loc[i, "turn"])
        ok_i = float(seq.loc[i, "ok"])

        if not in_chain:
            # 之前是 OK=1，当前变 0 -> 进入传播链
            # 或者“一上来就是 0”，也视为链从第一条 0 的 turn 开始
            if ok_i <= 0.5:
                in_chain = True
                start_turn = turn_i
        else:
            # 在链中，遇到 ok=1 表示恢复 -> 形成一条边
            if ok_i > 0.5:
                edges.append({
                    "src_turn": int(start_turn),
                    "dst_turn": int(turn_i),
                    "length": int(turn_i - start_turn),
                    "recovered": 1
                })
                in_chain = False
                start_turn = None

    # 序列结束仍在链中：以最后 turn 作为 dst（未恢复）
    if in_chain and start_turn is not None:
        last_turn = int(seq["turn"].iloc[-1])
        edges.append({
            "src_turn": int(start_turn),
            "dst_turn": int(last_turn),
            "length": int(last_turn - start_turn),
            "recovered": 0
        })

    return edges


def build_all_edges(ev: pd.DataFrame) -> pd.DataFrame:
    """
    对所有 (model, sample_id, img) 轨迹求传播边，拼成一个 DataFrame。
    """
    rows = []
    for (m, sid, img), g in ev.groupby(["model", "sample_id", "img"]):
        g = g.sort_values("turn")
        edges = find_propagation_edges(g)
        for e in edges:
            rows.append({
                "model": m,
                "sample_id": int(sid),
                "img": int(img),
                **e
            })
    out = pd.DataFrame(rows)
    if len(out) == 0:
        # 空保护
        return pd.DataFrame(columns=[
            "model", "sample_id", "img", "src_turn", "dst_turn", "length", "recovered"
        ])
    return out.sort_values(["model", "sample_id", "img", "src_turn"]).reset_index(drop=True)


def summarize_by_model(edges: pd.DataFrame) -> pd.DataFrame:
    """
    统计每个模型的传播特性：
    - n_edges：链条数
    - avg_len：平均长度
    - med_len：中位长度
    - recovery_rate：恢复比例（recovered=1 的占比）
    - long_chains(>=3)：长度≥3的比例（可调）
    """
    if len(edges) == 0:
        return pd.DataFrame(columns=[
            "model", "n_edges", "avg_len", "med_len", "recovery_rate", "long_chains_rate"
        ])

    def _safe_mean(s):
        return float(s.mean()) if len(s) else 0.0
    def _safe_median(s):
        return float(s.median()) if len(s) else 0.0

    stats = []
    for m, g in edges.groupby("model"):
        n = len(g)
        avg_len = _safe_mean(g["length"])
        med_len = _safe_median(g["length"])
        rec_rate = _safe_mean(g["recovered"])
        long_rate = float((g["length"] >= 3).mean()) if n > 0 else 0.0
        stats.append({
            "model": m,
            "n_edges": int(n),
            "avg_len": round(avg_len, 4),
            "med_len": round(med_len, 4),
            "recovery_rate": round(rec_rate, 4),
            "long_chains_rate": round(long_rate, 4)
        })
    out = pd.DataFrame(stats).sort_values("model").reset_index(drop=True)
    return out


def main():
    args = parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    events_path = os.path.join(in_dir, "events.tsv")
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"未找到 {events_path}；请先运行 build_stats_from_samples.py 生成 events.tsv")

    df = pd.read_csv(events_path, sep="\t")
    ev = explode_events(df)

    edges = build_all_edges(ev)
    edges_out = os.path.join(out_dir, "prop_edges.tsv")
    edges.to_csv(edges_out, sep="\t", index=False)
    print(f"[OK] saved {edges_out}")

    stats = summarize_by_model(edges)
    stats_out = os.path.join(out_dir, "propagation_stats.tsv")
    stats.to_csv(stats_out, sep="\t", index=False)
    print(f"[OK] saved {stats_out}")


if __name__ == "__main__":
    main()
