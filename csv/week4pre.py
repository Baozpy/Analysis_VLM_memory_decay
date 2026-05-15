import sys, subprocess
from pathlib import Path

ROOT  = Path(".").resolve()
BUILD = str(ROOT/"stats_from_arrays"/"build_stats_from_samples.py")
HALF  = str(ROOT/"stats_from_arrays"/"fit_half_life.py")
MIXED = str(ROOT/"stats_from_arrays"/"mixed_effects.py")
OUT   = ROOT/"real_stats"/"week4"
OUT.mkdir(parents=True, exist_ok=True)

def run(*args):
    print("$", *map(str, args))
    subprocess.check_call([sys.executable, *map(str, args)])

ks = [1,2,3,4,6,8,12,16]
for k in ks:
    arr = ROOT/"week4_ablations"/"variants_out"/f"recap_k{k}_fixed.py"
    out = OUT/f"recap_k{k}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n=== [k={k}] Step 1/3: build_stats_from_samples ===")
    run(BUILD, "--arrays", arr, "--out_dir", out)

    print(f"=== [k={k}] Step 2/3: fit_half_life ===")
    run(HALF, "--in-dir", out, "--out-dir", out)

    print(f"=== [k={k}] Step 3/3: mixed_effects ===")
    run(MIXED, "--in-dir", out, "--out-dir", out)

print("\n=== compare_week4 / summarize_week4 ===")
run(ROOT/"week4_ablations"/"compare_week4.py")
run(ROOT/"week4_ablations"/"summarize_week4.py")
print("== Done ==")
