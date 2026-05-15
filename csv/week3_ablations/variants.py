# -*- coding: utf-8 -*-
import random
from copy import deepcopy
from collections import defaultdict

def _permute_turns(pack, rng):
    T = len(pack["TURN_TO_IMAGES"])
    order = list(range(T))
    rng.shuffle(order)
    def perm(lst):
        return [lst[i] for i in order]
    out = deepcopy(pack)
    out["TURN_TO_IMAGES"] = perm(pack["TURN_TO_IMAGES"])
    for k in ("BLIP2","LLAVA","QWEN"):
        if pack[k] is not None:
            out[k] = perm(pack[k])
    return out

def rand_order(samples, seed=0):
    rng = random.Random(seed)
    out = {}
    for sid, pack in samples.items():
        out[sid] = _permute_turns(pack, rng)
    return out

def _group_by_image(turn_to_images):
    # 返回：每个 image_id -> 该图像出现的回合（按原顺序的切片）
    groups = defaultdict(list)
    for t_imgs in turn_to_images:
        for i in t_imgs:
            groups[i].append([i])  # 把多图回合拆解到单图片段
    return groups

def blocked(samples):
    # 把每个多图回合按“拆成若干单图回合”的近似方式重排成分块
    out = {}
    for sid, pack in samples.items():
        groups = _group_by_image(pack["TURN_TO_IMAGES"])
        # 顺序按图号从小到大
        new_tti = []
        for i in sorted(groups.keys()):
            new_tti.extend(groups[i])
        # 长度变化：为严格保持“回合数不变”，我们把分块后的 new_tti 修剪/填充到原长度
        T = len(pack["TURN_TO_IMAGES"])
        if len(new_tti) >= T:
            new_tti = new_tti[:T]
        else:
            # 若不够，追加最后一张图的若干回合
            while len(new_tti) < T and new_tti:
                new_tti.append(new_tti[-1])
            if not new_tti:
                new_tti = pack["TURN_TO_IMAGES"][:]  # 兜底
        out_pack = deepcopy(pack)
        out_pack["TURN_TO_IMAGES"] = new_tti
        out[sid] = out_pack
    return out

def interleave(samples):
    # 尽量轮转不同图像：抽取每张图的队列，round-robin 交替拼接
    out = {}
    for sid, pack in samples.items():
        # 准备每张图的“单图回合”队列
        per_img = defaultdict(list)
        for imgs in pack["TURN_TO_IMAGES"]:
            for i in imgs:
                per_img[i].append([i])
        img_ids = sorted(per_img.keys())
        queues = [per_img[i] for i in img_ids]
        new_tti = []
        idx = 0
        T = len(pack["TURN_TO_IMAGES"])
        while len(new_tti) < T and any(queues):
            q = queues[idx % len(queues)]
            if q:
                new_tti.append(q.pop(0))
            idx += 1
        # 长度不足则循环最后一个补齐
        while len(new_tti) < T and new_tti:
            new_tti.append(new_tti[-1])
        if not new_tti:
            new_tti = pack["TURN_TO_IMAGES"][:]
        out_pack = deepcopy(pack)
        out_pack["TURN_TO_IMAGES"] = new_tti
        out[sid] = out_pack
    return out

def shuffle_images(samples, seed=0):
    # 
    rng = random.Random(seed)
    out = {}
    for sid, pack in samples.items():
        # 
        uniq = sorted({i for imgs in pack["TURN_TO_IMAGES"] for i in imgs})
        perm = uniq[:]
        rng.shuffle(perm)
        mapping = {old: new for old, new in zip(uniq, perm)}
        new_tti = [[mapping[i] for i in imgs] for imgs in pack["TURN_TO_IMAGES"]]
        out_pack = deepcopy(pack)
        out_pack["TURN_TO_IMAGES"] = new_tti
        out[sid] = out_pack
    return out
