#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAR_DIR = Path(__file__).resolve().parent / "variants_out"
REAL = ROOT / "real_stats" / "week4"

BUILD = str(ROOT / "stats_from_arrays" / "build_stats_from_samples.py")
MIXED = str(ROOT / "stats_from_arrays" / "mixed_effects.py")
HALF  = str(ROOT / "stats_from_arrays" / "fit_half_life.py")

def sh(*args, cwd=None):
    print("$", *args)
    subprocess.check_call(args, cwd=cwd)

def run_one(name):
    out_dir = REAL / name
    out_dir.mkdir(parents=True, exist_ok=True)
    arr_py = VAR_DIR / f"{name}.py"
    sh("python", BUILD, "--arrays", str(arr_py), "--out_dir", str(out_dir), cwd=str(ROOT))
    sh("python", MIXED, "--in-dir", str(out_dir), "--out-dir", str(out_dir), cwd=str(ROOT))
    sh("python", HALF,  "--in-dir", str(out_dir), "--out-dir", str(out_dir), cwd=str(ROOT))

def main():
    REAL.mkdir(parents=True, exist_ok=True)
    variants = [p.stem for p in VAR_DIR.glob("*.py")]
    for v in variants:
        run_one(v)

if __name__ == "__main__":
    main()
