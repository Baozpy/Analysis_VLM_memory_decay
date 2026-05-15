#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从一个含 SAMPLE{sid}_TURN_TO_IMAGES / _BLIP2 / _LLAVA / _QWEN 的 .py 文件中
生成以下文件到 --out_dir：
  1) events.tsv                          （逐回合，含 sample_id/turn/images/model/ok）
  2) stats_delta_raw.tsv                 （逐图像逐回合：model, sample_id, image_id, turn, delta, ok）
  3) stats_k_raw.tsv                     （逐图像逐回合：model, sample_id, image_id, turn, k, ok）
  4) stats_tau_raw.tsv                   （逐图像逐回合：model, sample_id, image_id, turn, tau, ok）
  5) stats_Δ.tsv / stats_k.tsv / stats_τ.tsv （聚合版，含 ok_mean，向下兼容）

使用方式（与 Week3 runner 一致）：
  python build_stats_from_samples.py \
      --arrays path/to/variants_out/baseline_samples.py \
      --out_dir path/to/outdir
"""

import argparse, os, re
import pandas as pd

MODEL_KEYS = ("BLIP2", "LLAVA", "QWEN")
MODEL_NORM = { "BLIP2":"blip2", "LLAVA":"llava", "QWEN":"qwen" }
PAT = re.compile(r"^SAMPLE(\d+)_(TURN_TO_IMAGES|BLIP2|LLAVA|QWEN)$", re.I)

# build_stats_from_samples.py 内
import pathlib


def _load_arrays(pyfile: str):
    """
    兼容两种输入：
    A) 平铺变量：SAMPLE{i}_{BLIP2|LLAVA|QWEN|TURN_TO_IMAGES} = [...]
    B) 已打包：  PACKS = [{ "BLIP2":[...], "LLAVA":[...], "QWEN":[...], "TURN_TO_IMAGES":[...] }, ...]
    返回：Ordered dict: { sample_id(int): { "BLIP2":list[int], "LLAVA":..., "QWEN":..., ("TURN_TO_IMAGES":list[int])? } }
    """
    glb = {}
    code = pathlib.Path(pyfile).read_bytes()
    exec(compile(code, str(pyfile), "exec"), glb)

    # 情况B：已有 PACKS
    if "PACKS" in glb and isinstance(glb["PACKS"], (list, tuple)) and glb["PACKS"]:
        packs = {}
        for i, p in enumerate(glb["PACKS"]):
            if not isinstance(p, dict):
                continue
            packs[i] = {
                "BLIP2": [int(x) for x in p.get("BLIP2", [])],
                "LLAVA": [int(x) for x in p.get("LLAVA", [])],
                "QWEN":  [int(x) for x in p.get("QWEN",  [])],
            }
            if "TURN_TO_IMAGES" in p:
                packs[i]["TURN_TO_IMAGES"] = [int(x) for x in p["TURN_TO_IMAGES"]]
        if packs:
            return dict(sorted(packs.items()))

    # 情况A：平铺变量 SAMPLE{i}_*
    pat = re.compile(r'^SAMPLE(\d+)_(BLIP2|LLAVA|QWEN|TURN_TO_IMAGES)$')
    tmp = {}
    for k, v in glb.items():
        m = pat.match(k)
        if not m:
            continue
        sid = int(m.group(1))
        key = m.group(2)
        if not isinstance(v, (list, tuple)):
            continue
        tmp.setdefault(sid, {})[key] = [int(x) for x in v]

    # 至少要有任意一个模型数组
    packs = {}
    for sid, d in tmp.items():
        have_model = any(k in d for k in ("BLIP2","LLAVA","QWEN"))
        if not have_model:
            continue
        packs[sid] = {
            "BLIP2": d.get("BLIP2", []),
            "LLAVA": d.get("LLAVA", []),
            "QWEN":  d.get("QWEN",  []),
        }
        if "TURN_TO_IMAGES" in d:
            packs[sid]["TURN_TO_IMAGES"] = d["TURN_TO_IMAGES"]

    if not packs:
        raise RuntimeError("未在 --arrays 中发现任何 SAMPLE{i}_TURN_TO_IMAGES / *_BLIP2 / *_LLAVA / *_QWEN / PACKS")

    return dict(sorted(packs.items()))


def _first_mentions(turn_to_images):
    first_turn = {}
    for t, imgs in enumerate(turn_to_images, start=1):
        for i in imgs:
            if i not in first_turn:
                first_turn[i] = t
    return first_turn

def _revisit_k(turn_to_images):
    seen = {}
    k_at = {}
    for t, imgs in enumerate(turn_to_images, start=1):
        for i in imgs:
            k_at[(t, i)] = seen.get(i, 0)  # 首次出现 k=0
            seen[i] = seen.get(i, 0) + 1
    return k_at

def _build_rows_for_sample(sid, tti, ok_by_model):
    """
    产出 per-image per-turn per-model 的事件行。
    必含列：sample_id, turn, images(逗号串), model, ok, image_id, delta
    - image_id: 当前行对应的单个图像 ID；若该回合无图像，用 -1 占位
    - delta   : t - origin_turn(image_id)；若无图像/未知，置 None
    """
    rows = []
    # 1) 预计算每个图像的“首次出现回合”——用于 delta = 当前回合 - 首次回合
    origin_turn = {}
    for t, imgs in enumerate(tti):
        if not isinstance(imgs, (list, tuple)):
            continue
        for img in imgs:
            if img is None:
                continue
            if img not in origin_turn:
                origin_turn[int(img)] = t

    T = len(tti)
    for t in range(T):
        imgs = tti[t] if isinstance(tti[t], (list, tuple)) else []
        # 用于 events.tsv 的整回合图像串
        images_str = ",".join(str(int(x)) for x in imgs) if imgs else ""

        # 如果该回合没有图像，也要产出占位行（保证列完整）
        per_turn_imgs = list(imgs) if imgs else [None]

        for model, oks in ok_by_model.items():
            # 跳过不合法或长度不够的 ok 列表
            if not isinstance(oks, list) or t >= len(oks):
                continue
            ok_t = int(oks[t])

            for img in per_turn_imgs:
                if img is None:
                    image_id = -1
                    delta = None
                else:
                    image_id = int(img)
                    # 若该图像从未在 origin_turn 记录（理论上不会），则置 None
                    delta = t - origin_turn.get(image_id, t)  # 没找到就当本回合首次，delta=0

                rows.append({
                    "sample_id": int(sid),
                    "turn": t,                 # 如需 1-based，这里改成 t+1，但要与历史一致
                    "images": images_str,      # 本回合的全图像串，空回合为 ""
                    "model": model,            # 'blip2' / 'llava' / 'qwen'
                    "ok": ok_t,                # 0/1
                    "image_id": image_id,      # 单个图像ID；无图像=-1
                    "delta": delta,            # None 或 非负整数
                })
    return rows



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", required=True, help="含 SAMPLE{sid}_... 的 .py")
    ap.add_argument("--out_dir", required=True, help="输出目录")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    samples = _load_arrays(args.arrays)
    if not samples:
        raise RuntimeError("未在 --arrays 中发现任何 SAMPLE{i}_TURN_TO_IMAGES / *_BLIP2 / *_LLAVA / *_QWEN")

    # all_rows = []
    # for sid, pack in sorted(samples.items()):
    #     tti = pack["TURN_TO_IMAGES"]
    #     ok_by_model = {k: pack.get(k) for k in MODEL_KEYS}
    #     all_rows += _build_rows_for_sample(sid, tti, ok_by_model)
    # samples = _load_arrays(args.arrays)  # {sid: pack}
    # all_rows = []

    # … 你的前置代码 …
    # 读取所有样本
    samples = _load_arrays(args.arrays)  # {sid: pack}

    # ---- 解析变体里的 k，并用于裁剪回合 ----
    import re
    m = re.search(r"recap_k(\d+)", os.fspath(args.arrays))
    _k_from_name = int(m.group(1)) if m else None
    if _k_from_name:
        print(f"[clip] detected variant k={_k_from_name} from {args.arrays}")

    all_rows = []
    for sid in sorted(samples.keys()):
        pack = samples[sid]

        # 1) 取 TURN_TO_IMAGES（可能不存在）
        tti = pack.get("TURN_TO_IMAGES")
        # 2) 取模型序列
        ok_by_model = {k: (pack.get(k) or []) for k in MODEL_KEYS}

        # === 按 k 裁剪（若检测到 k）===
        if _k_from_name:
            # 裁剪模型 ok 序列
            for mk in MODEL_KEYS:
                if isinstance(ok_by_model.get(mk), list):
                    ok_by_model[mk] = ok_by_model[mk][:_k_from_name]
            # 裁剪 TURN_TO_IMAGES
            if isinstance(tti, list):
                tti = tti[:_k_from_name]

        # === 若没有 tti，则根据最长模型序列长度生成一个占位的 [[] * T] ===
        if tti is None:
            lengths = [len(v) for v in ok_by_model.values() if isinstance(v, list)]
            T = max(lengths) if lengths else 0
            tti = [[] for _ in range(T)]

        # 生成逐回合行
        all_rows += _build_rows_for_sample(sid, tti, ok_by_model)

    # 落成 DataFrame
    df = pd.DataFrame(all_rows)

    # === 填补列：k / image_id / delta ===
    if "k" not in df.columns:
        df["k"] = _k_from_name if _k_from_name is not None else pd.NA

    if "image_id" not in df.columns:
        df["image_id"] = -1

    if "delta" not in df.columns and "turn" in df.columns:
        df["delta"] = df["turn"]


    # ===== 填补缺列：k / image_id / delta / images（必要时） =====
    m = re.search(r"recap_k(\d+)", os.fspath(args.arrays))
    if "k" not in df.columns:            df["k"] = int(m.group(1)) if m else pd.NA
    if "image_id" not in df.columns:     df["image_id"] = -1
    if "delta" not in df.columns and "turn" in df.columns:
                                        df["delta"] = df["turn"]
    if "images" not in df.columns:       df["images"] = ""   # ✅ 防止 events 选列报错

    # ===== 只在存在 tau 时写 tau 表 =====
    from pathlib import Path
    out_path = Path(args.out_dir)

    def _maybe_write_tau_tables(df, out_dir: Path):
        if "tau" not in df.columns:
            print("[warn] skip tau tables: 'tau' column missing")
            return
        stats_tau_raw = df[["model","sample_id","image_id","turn","tau","ok"]].sort_values(
            ["model","sample_id","turn","image_id","tau"]
        )
        stats_tau_raw.to_csv(out_dir / "stats_tau_raw.tsv", sep="\t", index=False)
        stats_tau = (
            stats_tau_raw.groupby(["model","tau"], as_index=False)["ok"]
            .mean().rename(columns={"ok":"ok_mean"})
            .sort_values(["model","tau"])
        )
        stats_tau.to_csv(out_dir / "stats_τ.tsv", sep="\t", index=False)

    # ===== 其余 raw 表 =====
    events = df[["sample_id","turn","images","model","ok"]].drop_duplicates()
    events.to_csv(os.path.join(args.out_dir, "events.tsv"), sep="\t", index=False)

    stats_delta_raw = df[["model","sample_id","image_id","turn","delta","ok"]].sort_values(
        ["model","sample_id","image_id","turn"]
    )
    stats_delta_raw.to_csv(os.path.join(args.out_dir, "stats_delta_raw.tsv"), sep="\t", index=False)

    stats_k_raw = df[["model","sample_id","image_id","turn","k","ok"]].sort_values(
        ["model","sample_id","image_id","turn"]
    )
    stats_k_raw.to_csv(os.path.join(args.out_dir, "stats_k_raw.tsv"), sep="\t", index=False)

    # ✅ 只通过这个函数来（有条件地）写 tau 表；不要再有无条件的 tau 输出
    _maybe_write_tau_tables(df, out_path)

    print("[OK] wrote raw tables to", args.out_dir)




    # for sid in sorted(samples.keys()):
    #     pack = samples[sid]
    #     tti = pack.get("TURN_TO_IMAGES")

    #     # 没有 tti 时，用模型序列长度推一个空图列表
    #     if tti is None:
    #         lengths = [len(v) for k, v in pack.items() if k in MODEL_KEYS and isinstance(v, list)]
    #         T = max(lengths) if lengths else 0
    #         tti = [[] for _ in range(T)]

    #     ok_by_model = {k: (pack.get(k) or []) for k in MODEL_KEYS}
    #     all_rows += _build_rows_for_sample(sid, tti, ok_by_model)

   
   
    # df = pd.DataFrame(all_rows)
    # # ===== 填补缺列：k / image_id / delta（必要时） =====
    # # 1) k: 从 --arrays 文件名里解析 recap_kX
    # if "k" not in df.columns:
    #     m = re.search(r"recap_k(\d+)", os.fspath(args.arrays))
    #     if m:
    #         df["k"] = int(m.group(1))
    #     else:   
    #         df["k"] = pd.NA  # 实在解析不到就置空

    # # 2) image_id: 若没有 TURN_TO_IMAGES，我们用 -1 占位，保证下游 groupby 能跑
    # if "image_id" not in df.columns:
    #     df["image_id"] = -1

    # # 3) delta: 若缺失，用 turn 兜底（half-life 仍可按“相对回合”拟合）
    # if "delta" not in df.columns and "turn" in df.columns:
    #     df["delta"] = df["turn"]

    # # ===== 可选导出 tau：缺列就跳过 =====
    # def _maybe_write_tau_tables(df, out_dir):
    #     if "tau" not in df.columns:
    #         print("[warn] skip tau tables: 'tau' column missing")
    #         return
    #     stats_tau_raw = df[["model","sample_id","image_id","turn","tau","ok"]].sort_values(
    #         ["model","sample_id","turn","image_id","tau"]
    #     )
    #     stats_tau_raw.to_csv(out_dir / "stats_tau_raw.tsv", sep="\t", index=False)
    #     stats_tau = (
    #         stats_tau_raw.groupby(["model","tau"], as_index=False)["ok"]
    #         .mean()
    #         .rename(columns={"ok":"ok_mean"})
    #         .sort_values(["model","tau"])
    #     )
    #     stats_tau.to_csv(out_dir / "stats_τ.tsv", sep="\t", index=False)

    # # ……你现有的 events / stats_delta_raw / stats_k_raw 等写完之后：
    # # 调用一下：
    

    

    # #===============================================
    # # 1) events.tsv（逐回合去重）
    # events = df[["sample_id","turn","images","model","ok"]].drop_duplicates()
    # events.to_csv(os.path.join(args.out_dir, "events.tsv"), sep="\t", index=False)
    # print("[OK] wrote", os.path.join(args.out_dir, "events.tsv"))

    # # 2) 原始逐图像表（raw）
    # stats_delta_raw = df[["model","sample_id","image_id","turn","delta","ok"]].sort_values(
    #     ["model","sample_id","image_id","turn"]
    # )
    # stats_delta_raw.to_csv(os.path.join(args.out_dir, "stats_delta_raw.tsv"), sep="\t", index=False)

    # stats_k_raw = df[["model","sample_id","image_id","turn","k","ok"]].sort_values(
    #     ["model","sample_id","image_id","turn"]
    # )
    # stats_k_raw.to_csv(os.path.join(args.out_dir, "stats_k_raw.tsv"), sep="\t", index=False)

    # stats_tau_raw = df[["model","sample_id","image_id","turn","tau","ok"]].sort_values(
    #     ["model","sample_id","image_id","turn"]
    # )
    # stats_tau_raw.to_csv(os.path.join(args.out_dir, "stats_tau_raw.tsv"), sep="\t", index=False)

    # _maybe_write_tau_tables(df, out_dir)

    # print("[OK] wrote raw tables to", args.out_dir)

    # 3) 向下兼容的聚合版（与旧脚本一致：只有 model / 指标 / ok_mean）

    # ===== 向下兼容的聚合（Δ / k 固定写；τ 交给上面的函数去负责） =====
    agg_delta = (
        stats_delta_raw.groupby(["model", "delta"], as_index=False)["ok"]
        .mean()
        .rename(columns={"ok": "ok_mean"})
        .sort_values(["model", "delta"])
    )
    agg_delta.to_csv(os.path.join(args.out_dir, "stats_Δ.tsv"), sep="\t", index=False)

    agg_k = (
        stats_k_raw.groupby(["model", "k"], as_index=False)["ok"]
        .mean()
        .rename(columns={"ok": "ok_mean"})
        .sort_values(["model", "k"])
    )
    agg_k.to_csv(os.path.join(args.out_dir, "stats_k.tsv"), sep="\t", index=False)

    # τ 的聚合（stats_τ.tsv）如果存在，会在 _maybe_write_tau_tables 里生成；
    # 这里不要再引用 stats_tau_raw 以免在没有 tau 时 NameError
    if "tau" in df.columns:
        print("[OK] also wrote stats_τ.tsv via _maybe_write_tau_tables")
    else:
        print("[warn] skip stats_τ.tsv: 'tau' column missing")

    print("[OK] wrote stats_Δ.tsv / stats_k.tsv to", args.out_dir)


if __name__ == "__main__":
    main()
