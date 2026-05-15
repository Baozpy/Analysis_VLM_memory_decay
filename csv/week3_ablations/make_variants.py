#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从一个“源样本文件”（含 SAMPLE{sid}_TURN_TO_IMAGES / _BLIP2 / _LLAVA / _QWEN 全量数组）
生成多种 Week3 结构扰动变体，并写到 variants_out/ 下为 .py 文件。
每个变体严格输出同名全局变量：SAMPLE{sid}_TURN_TO_IMAGES / _BLIP2 / _LLAVA / _QWEN
从而确保 downstream 的 build_stats_from_samples.py 能识别出 sample_id。

用法：
    python week3_ablations/make_variants.py \
        --source build_stats_samples.py \
        --out-dir week3_ablations/variants_out \
        --seeds 0,1,2
"""

import argparse, os, re, random, textwrap
from copy import deepcopy

MODEL_KEYS = ("BLIP2", "LLAVA", "QWEN")
PAT = re.compile(r"^SAMPLE(\d+)_(TURN_TO_IMAGES|BLIP2|LLAVA|QWEN)$", re.I)

def load_samples(pyfile: str):
    glb = {}
    with open(pyfile, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, pyfile, "exec"), glb)
    # 收集 SAMPLE*
    tmp = {}
    for name, val in glb.items():
        if not isinstance(name, str): 
            continue
        m = PAT.match(name)
        if not m:
            continue
        sid = int(m.group(1))
        key = m.group(2).upper()
        pack = tmp.setdefault(sid, {"TURN_TO_IMAGES": None, "BLIP2": None, "LLAVA": None, "QWEN": None})
        pack[key] = val
    # 过滤至少有 TURN_TO_IMAGES 且至少一个模型
    samples = {}
    for sid, pack in tmp.items():
        if pack["TURN_TO_IMAGES"] is None:
            continue
        if any(pack[k] is not None for k in MODEL_KEYS):
            samples[sid] = pack
    return samples

def write_samples_py(path: str, samples: dict, header_title: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# -*- coding: utf-8 -*-\n")
        f.write(f"# {header_title}\n\n")
        # 逐个 SAMPLE 写四类全局变量
        for sid in sorted(samples.keys()):
            pack = samples[sid]
            # TURN_TO_IMAGES
            f.write(f"SAMPLE{sid}_TURN_TO_IMAGES = {repr(pack['TURN_TO_IMAGES'])}\n")
            # 模型 OK 序列（允许 None）
            for k in MODEL_KEYS:
                key = f"SAMPLE{sid}_{k}"
                f.write(f"{key} = {repr(pack.get(k))}\n")
            f.write("\n")

def reorder_turns(turn_to_images, ok_by_model, order):
    """按照给定的 turn 顺序重排 TTI 与 OK。"""
    T = len(turn_to_images)
    assert sorted(order) == list(range(T)), "order 必须是 0..T-1 的置换"
    new_tti = [turn_to_images[i] for i in order]
    new_ok = {}
    for m, seq in ok_by_model.items():
        if seq is None:
            new_ok[m] = None
        else:
            new_ok[m] = [seq[i] for i in order]
    return new_tti, new_ok

def permute_image_ids(turn_to_images, perm):
    """对每个回合的图号做置换（回合顺序与 OK 不变）。"""
    def map_one(imgs):
        return [perm.get(i, i) for i in imgs]
    return [map_one(imgs) for imgs in turn_to_images]

# === 变体策略 ===

def variant_baseline(samples):
    return deepcopy(samples)

def variant_rand_order(samples, seed=0):
    rnd = random.Random(seed)
    out = {}
    for sid, pack in samples.items():
        T = len(pack["TURN_TO_IMAGES"])
        order = list(range(T))
        rnd.shuffle(order)
        new_tti, new_ok = reorder_turns(pack["TURN_TO_IMAGES"],
                                        {k: pack.get(k) for k in MODEL_KEYS},
                                        order)
        out[sid] = dict(TURN_TO_IMAGES=new_tti, **new_ok)
    return out

def variant_blocked(samples):
    """把相同 image 的回合尽量聚在一起（简单贪心：按首次出现的 image 序列分块）。"""
    out = {}
    for sid, pack in samples.items():
        tti = pack["TURN_TO_IMAGES"]
        # 找到每张图的首次出现 turn
        first_turn = {}
        for t, imgs in enumerate(tti):
            for i in imgs:
                if i not in first_turn:
                    first_turn[i] = t
        images_in_order = sorted(first_turn.keys(), key=lambda i: first_turn[i])
        # 分块：把包含该 image 的回合先收集，避免重复
        T = len(tti)
        used = [False]*T
        order = []
        for img in images_in_order:
            for t, imgs in enumerate(tti):
                if not used[t] and img in imgs:
                    order.append(t)
                    used[t] = True
        # 追加剩余未用回合
        for t in range(T):
            if not used[t]:
                order.append(t)
                used[t] = True
        new_tti, new_ok = reorder_turns(tti, {k:pack.get(k) for k in MODEL_KEYS}, order)
        out[sid] = dict(TURN_TO_IMAGES=new_tti, **new_ok)
    return out

def variant_interleave(samples):
    """尽量交替不同 image（简单轮转：按回合中出现的“主图”轮换）。"""
    out = {}
    for sid, pack in samples.items():
        tti = pack["TURN_TO_IMAGES"]
        # 取每回合的“主图”（多图回合取最小 id，足够稳定）
        main = [min(imgs) for imgs in tti]
        # 目标顺序：尽量让相邻 main 不同：按 main 的值做桶，再 round-robin 取
        from collections import defaultdict, deque
        buckets = defaultdict(deque)
        for idx, m in enumerate(main):
            buckets[m].append(idx)
        keys = sorted(buckets.keys())
        order = []
        while any(buckets[k] for k in keys):
            for k in keys:
                if buckets[k]:
                    order.append(buckets[k].popleft())
        new_tti, new_ok = reorder_turns(tti, {k:pack.get(k) for k in MODEL_KEYS}, order)
        out[sid] = dict(TURN_TO_IMAGES=new_tti, **new_ok)
    return out

def variant_shuffle_images(samples, seed=0):
    """仅置换图号，不改回合顺序与 OK。对每个 SAMPLE 独立生成置换。"""
    rnd = random.Random(seed)
    out = {}
    for sid, pack in samples.items():
        tti = pack["TURN_TO_IMAGES"]
        # 收集出现过的 image id
        S = sorted({i for imgs in tti for i in imgs})
        perm_keys = S[:]
        rnd.shuffle(perm_keys)
        perm = dict(zip(S, perm_keys))
        new_tti = permute_image_ids(tti, perm)
        # OK 不变
        new_ok = {k: deepcopy(pack.get(k)) for k in MODEL_KEYS}
        out[sid] = dict(TURN_TO_IMAGES=new_tti, **new_ok)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="含 SAMPLE{sid}_... 的源 .py")
    ap.add_argument("--out-dir", required=True, help="输出目录")
    ap.add_argument("--seeds", default="0,1,2", help="随机种子列表，逗号分隔，用于 rand_order / shuffle_images")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()!=""]
    os.makedirs(args.out_dir, exist_ok=True)

    base = load_samples(args.source)

    # baseline
    write_samples_py(os.path.join(args.out_dir, "baseline_samples.py"),
                     variant_baseline(base), "Baseline (unchanged)")

    # rand_order for each seed
    for s in seeds:
        write_samples_py(os.path.join(args.out_dir, f"rand_order_s{s}.py"),
                         variant_rand_order(base, seed=s), f"Random turn order (seed={s})")

    # blocked
    write_samples_py(os.path.join(args.out_dir, "blocked.py"),
                     variant_blocked(base), "Blocked by image")

    # interleave
    write_samples_py(os.path.join(args.out_dir, "interleave.py"),
                     variant_interleave(base), "Interleave images")

    # shuffle_images for each seed
    for s in seeds:
        write_samples_py(os.path.join(args.out_dir, f"shuffle_images_s{s}.py"),
                         variant_shuffle_images(base, seed=s), f"Shuffle image IDs (seed={s})")

    print(f"[OK] Variants written to {args.out_dir}")

if __name__ == "__main__":
    main()
