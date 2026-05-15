# controlled/runner_stub.py
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent       # .../CSV/controlled
CSVROOT = HERE.parent                        # .../CSV
STATS_DIR = CSVROOT / "stats_from_arrays"

def sh(*args, cwd=None):
    print("$", *args)
    subprocess.check_call(args, cwd=cwd)

def main():
    # 1) 
    sh("python", "generate_controlled.py", cwd=HERE)

    # 2) 
    build_script = CSVROOT / "build_stats_from_samples.py"
    if not build_script.exists():
        # 
        build_script = STATS_DIR / "build_stats_from_samples.py"
    sh(
        "python", str(build_script),
        "--arrays", str(HERE / "controlled_samples.py"),
        "--out_dir", str(STATS_DIR),
        cwd=CSVROOT if build_script.parent == CSVROOT else STATS_DIR
    )

    # 3) 
    sh("python", "mixed_effects.py", cwd=STATS_DIR)
    sh("python", "fit_half_life.py", cwd=STATS_DIR)
    sh("python", "build_propagation.py", cwd=STATS_DIR)

if __name__ == "__main__":
    main()
