# -*- coding: utf-8 -*-
import os, subprocess, glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATS_BUILD = os.path.join(ROOT, "stats_from_arrays", "build_stats_from_samples.py")
MIXED = os.path.join(ROOT, "stats_from_arrays", "mixed_effects.py")
HALF  = os.path.join(ROOT, "stats_from_arrays", "fit_half_life.py")
PROP  = os.path.join(ROOT, "stats_from_arrays", "build_propagation.py")

VAR_DIR = os.path.join(ROOT, "week3_ablations", "variants_out")  # make_variants.py 生成到这
OUT_ROOT = os.path.join(ROOT, "real_stats", "week3")

def sh(*args, cwd=None):
    print("$", *args)
    subprocess.check_call(args, cwd=cwd)

def run_one(name, arrays_py):
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    # 1) per-image 事件
    sh("python", STATS_BUILD, "--arrays", arrays_py, "--out_dir", out_dir, cwd=ROOT)
    # 2) 三个二阶段统计
    sh("python", MIXED, "--in-dir", out_dir, "--out-dir", out_dir, cwd=ROOT)
    sh("python", HALF,  "--in-dir", out_dir, "--out-dir", out_dir, cwd=ROOT)
    sh("python", PROP,  "--in-dir", out_dir, "--out-dir", out_dir, cwd=ROOT)

def main():
    # 先确保你已运行：
    #   python week3_ablations/make_variants.py --source build_stats_samples.py --out-dir week3_ablations/variants_out --seeds 0,1,2
    variant_files = sorted(glob.glob(os.path.join(VAR_DIR, "*.py")))
    for vf in variant_files:
        name = os.path.splitext(os.path.basename(vf))[0]
        run_one(name, vf)

if __name__ == "__main__":
    main()
