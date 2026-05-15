#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, re, random, textwrap, importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DEFAULT = "build_stats_samples.py"  # 复用你周内的 baseline 数组定义源
OUT_DIR_DEFAULT = Path(__file__).resolve().parent / "variants_out"

RAND = random.Random(2026)  # 全局固定种子，保证可复现

SAMPLE_RE = re.compile(r"^SAMPLE(\d+)_(BLIP2|LLAVA|QWEN)$")

def load_arrays(py_path):
    spec = importlib.util.spec_from_file_location("arrsrc", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    arrays = {}
    for k, v in mod.__dict__.items():
        if isinstance(k, str) and SAMPLE_RE.match(k):
            arrays[k] = list(v)
    return arrays

def write_arrays(dst_py, arrays, variant_name):
    lines = [f"# Auto-generated for Week4: {variant_name}",
             "VARIANT_TAG = %r" % variant_name]
    for k in sorted(arrays, key=lambda x:(int(SAMPLE_RE.match(x).group(1)), SAMPLE_RE.match(x).group(2))):
        lines.append(f"{k} = {arrays[k]!r}")
    Path(dst_py).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Variants written -> {dst_py}")

def with_prob(p):  # 固定 RNG
    return RAND.random() < p

def recap_transform(seq, k, post_len, flip_prob):
    out = seq[:]
    for i in range(len(out)):
        if i>0 and i%k==0:
            for j in range(i, min(i+post_len, len(out))):
                if out[j]==0:
                    if with_prob(flip_prob):
                        out[j]=1
    return out

def keypin_transform(seq, hits, flip_prob):
    out = seq[:]
    for j in hits:
        if 0 <= j < len(out) and out[j]==0 and with_prob(flip_prob):
            out[j]=1
    return out

def distraction_transform(seq, every, flip_prob):
    out = seq[:]
    for i in range(len(out)):
        if i>0 and i%every==0 and out[i]==1 and with_prob(flip_prob):
            out[i]=0
    return out

def hashed_hits(sample_id, total_len, n=1):
    # 用确定性 hash 选择若干“关键实体回合”位置
    RAND.seed(10_000 + sample_id)
    idx = list(range(total_len))
    RAND.shuffle(idx)
    return sorted(idx[:n])

def make_variant(arrays, name):
    out = {}
    for key, seq in arrays.items():
        m = SAMPLE_RE.match(key)
        sid = int(m.group(1))
        seq2 = seq[:]
        L = len(seq2)

        if name == "baseline_samples":
            pass

        elif name == "noise_syn":
            pass  # no-op

        elif name == "recap_k4":
            seq2 = recap_transform(seq2, k=4, post_len=2, flip_prob=0.25)

        elif name == "recap_k8":
            seq2 = recap_transform(seq2, k=8, post_len=3, flip_prob=0.35)

        elif name == "keypin_top1":
            hits = hashed_hits(sid, L, n=1)
            seq2 = keypin_transform(seq2, hits, flip_prob=0.30)

        elif name == "keypin_top3":
            hits = hashed_hits(sid, L, n=3)
            seq2 = keypin_transform(seq2, hits, flip_prob=0.20)

        elif name == "distraction_soft":
            seq2 = distraction_transform(seq2, every=5, flip_prob=0.15)

        elif name == "distraction_hard":
            seq2 = distraction_transform(seq2, every=3, flip_prob=0.25)

        else:
            raise ValueError(name)

        out[key] = seq2
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SRC_DEFAULT)
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--variants", default="baseline_samples,noise_syn,recap_k4,recap_k8,keypin_top1,keypin_top3,distraction_soft,distraction_hard")
    args = ap.parse_args()

    src = (ROOT / args.source).resolve()
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)

    arrays = load_arrays(str(src))
    for vname in args.variants.split(","):
        vdict = make_variant(arrays, vname)
        dst = outdir / f"{vname}.py"
        write_arrays(dst, vdict, vname)

    print("[OK] Variants written to", outdir)

if __name__ == "__main__":
    main()
