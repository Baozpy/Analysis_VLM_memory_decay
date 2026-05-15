# -*- coding: utf-8 -*-
"""
Generate controlled samples and write controlled_samples.py for stats scripts.
No external imports from runner_stub. Purely synthetic (controlled) labels.
"""

from pathlib import Path
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent                  # .../CSV/controlled
MANUAL = ROOT.parent / "manual_plots"                  # .../CSV/manual_plots
SAMPLES_OUT = ROOT / "controlled_samples.py"           # output arrays here

MANUAL.mkdir(parents=True, exist_ok=True)

# ---------- Controlled spec ----------
random.seed(2026)

@dataclass
class ControlledSample:
    turns_to_images: List[List[int]]           # e.g., [[1],[1,2],...]
    ok_blip2: List[int]
    ok_llava: List[int]
    ok_qwen:  List[int]

@dataclass
class Design:
    T: int = 20
    images: int = 4
    # event schedule: first-mention turns for i=1..images
    first_mentions: Dict[int, int] = field(default_factory=lambda: {1:1, 2:4, 3:7, 4:10})

def make_turn_to_images(des: Design) -> List[List[int]]:
    """Simple controlled pattern: each image has 3 dedicated turns after first mention,
    then a few mixed-image turns later."""
    T = des.T
    f = des.first_mentions
    tti: List[List[int]] = []
    for t in range(1, T+1):
        if t in [f[1], f[1]+1, f[1]+2]:
            tti.append([1])
        elif t in [f[2], f[2]+1, f[2]+2]:
            tti.append([2])
        elif t in [f[3], f[3]+1, f[3]+2]:
            tti.append([3])
        elif t in [f[4], f[4]+1, f[4]+2]:
            tti.append([4])
        else:
            # mixed turns after everyone has been mentioned at least once
            if t > max(f.values()):
                # simple rotating mixes
                mixes = [[1,4], [2,3], [1,2,4], [3,4], [1,2], [2,3], [1,3,4]]
                tti.append(mixes[(t - max(f.values()) - 1) % len(mixes)])
            else:
                # idle filler before all first-mentions complete -> default to first image
                tti.append([1])
    return tti

# ---------- Probability gadgets for synthetic labels ----------

def b2_prob(delta: int, model: str) -> float:
    """Answer-OK probability as a function of Δ (turns since first mention)."""
    # hand-crafted curves similar to你的 pilot 图
    base = {"BLIP2": 0.28, "LLaVA": 0.60, "Qwen": 0.67}[model]
    decay_step = {"BLIP2": 0.015, "LLaVA": 0.012, "Qwen": 0.010}[model]
    # slight valley near Δ≈6 then rebound:
    bump = -0.10 if delta >= 6 else 0.0
    val = base - decay_step * delta + bump
    return max(0.05, min(0.95, val))

def revisit_index(mentions: List[int], t: int) -> int:
    """k-th revisit index for the image asked at turn t."""
    # mentions is a sorted list of prior turns this image has appeared (including first)
    return len([x for x in mentions if x <= t])

def d_prob(k: int, model: str) -> float:
    """Revisit curve probability P(ok|k)."""
    tops = {"BLIP2": 0.27, "LLaVA": 0.61, "Qwen": 0.69}[model]
    if k <= 3:
        return tops - 0.02*(k-1)
    if k == 4:
        return {"BLIP2": 0.08, "LLaVA": 0.29, "Qwen": 0.34}[model]
    if k == 5:
        return {"BLIP2": 0.12, "LLaVA": 0.42, "Qwen": 0.46}[model]
    return {"BLIP2": 0.115, "LLaVA": 0.37, "Qwen": 0.49}[model]

def b1_prob(t: int, T: int, model: str) -> float:
    """Normalized dialogue progress τ = t/T → prob."""
    tau = t / T
    # early better, later worse; Qwen highest
    start = {"BLIP2": 0.30, "LLaVA": 0.66, "Qwen": 0.77}[model]
    end   = {"BLIP2": 0.16, "LLaVA": 0.43, "Qwen": 0.47}[model]
    val = start + (end - start) * (tau ** 0.9)
    return max(0.05, min(0.95, val))

def bernoulli(p: float) -> int:
    return 1 if random.random() < p else 0

def synthesize_ok_arrays(tti: List[List[int]], des: Design, model: str) -> List[int]:
    """Combine three signals (B2, D, B1) into a single per-turn OK label."""
    # track first mention turn per image and revisit counts
    first: Dict[int, int] = {}
    mentions: Dict[int, List[int]] = {i:[] for i in range(1, des.images+1)}
    ok: List[int] = []
    for t, imgs in enumerate(tti, start=1):
        # choose a representative image for multi-image turns: take the earliest-first-mention one
        rep = min(imgs, key=lambda i: first.get(i, 10**9))
        if rep not in first:
            first[rep] = t
        mentions[rep].append(t)

        Δ = t - first[rep]
        k = revisit_index(mentions[rep], t)
        p = 0.45 * b2_prob(Δ, model) + 0.30 * d_prob(k, model) + 0.25 * b1_prob(t, des.T, model)
        ok.append(bernoulli(p))
    return ok

def one_controlled_sample(idx: int) -> ControlledSample:
    des = Design()
    tti = make_turn_to_images(des)
    return ControlledSample(
        turns_to_images = tti,
        ok_blip2 = synthesize_ok_arrays(tti, des, "BLIP2"),
        ok_llava = synthesize_ok_arrays(tti, des, "LLaVA"),
        ok_qwen  = synthesize_ok_arrays(tti, des, "Qwen"),
    )

# ---------- writer ----------
HEADER = """# Auto-generated controlled samples
# Each SAMPLE{n}_TURN_TO_IMAGES / _BLIP2 / _LLAVA / _QWEN
"""

def dump_samples(samples: List[ControlledSample], path: Path):
    with path.open("w", encoding="utf-8") as f:
        f.write(HEADER)
        for n, s in enumerate(samples, start=1):
            f.write(f"SAMPLE{n}_TURN_TO_IMAGES = {s.turns_to_images}\n")
            f.write(f"SAMPLE{n}_BLIP2 = {s.ok_blip2}\n")
            f.write(f"SAMPLE{n}_LLAVA = {s.ok_llava}\n")
            f.write(f"SAMPLE{n}_QWEN  = {s.ok_qwen}\n\n")
    print(f"[OK] Wrote {len(samples)} controlled samples -> {path}")

def main():
    # 
    samples = [one_controlled_sample(i) for i in range(1, 6)]
    dump_samples(samples, SAMPLES_OUT)

if __name__ == "__main__":
    main()
