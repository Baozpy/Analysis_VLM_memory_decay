# -*- coding: utf-8 -*-
import os, pandas as pd, glob

def load_half(dirpath):
    fp = os.path.join(dirpath, "half_life.tsv")
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, sep="\t")
    df["variant"] = os.path.basename(dirpath)
    return df

def load_mixed(dirpath):
    fp = os.path.join(dirpath, "mixed_effects.tsv")
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, sep="\t")
    df["variant"] = os.path.basename(dirpath)
    return df

def load_prop(dirpath):
    fp = os.path.join(dirpath, "propagation_stats.tsv")
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, sep="\t")
    df["variant"] = os.path.basename(dirpath)
    return df

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "real_stats", "week3"))
    subdirs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
    half = [load_half(d) for d in subdirs]; half = [x for x in half if x is not None]
    mixed = [load_mixed(d) for d in subdirs]; mixed = [x for x in mixed if x is not None]
    prop = [load_prop(d) for d in subdirs]; prop = [x for x in prop if x is not None]

    os.makedirs(root, exist_ok=True)
    if half:
        pd.concat(half, ignore_index=True).to_csv(os.path.join(root, "Z_compare_half_life.tsv"), sep="\t", index=False)
    if mixed:
        pd.concat(mixed, ignore_index=True).to_csv(os.path.join(root, "Z_compare_mixed_effects.tsv"), sep="\t", index=False)
    if prop:
        pd.concat(prop, ignore_index=True).to_csv(os.path.join(root, "Z_compare_propagation.tsv"), sep="\t", index=False)
    print("[OK] Wrote Z_compare_*.tsv in", root)

if __name__ == "__main__":
    main()
