# plot_decay.py
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def load_focus_curve(path: str, label: str):
    """
    从 *_analysis.csv 里读 turn 和 focus_correct
    返回 turn_list, score_list, label
    """
    df = pd.read_csv(path)
    # 有些 turn 问题不提具体ImageK -> focus_correct = NaN
    # 我们在画图时可以保留，也可以把 NaN 当成前一回合的值或直接跳
    # 先简单策略：把 NaN 当成上一轮的值；第一轮如果NaN就当0.0
    turns = df["turn"].tolist()
    vals_raw = df["focus_correct"].tolist()

    vals_filled = []
    last_val = 0.0
    for v in vals_raw:
        if pd.isna(v):
            vals_filled.append(last_val)
        else:
            last_val = float(v)
            vals_filled.append(last_val)

    return turns, vals_filled, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="形如 modelName=path/to/analysis.csv，多模型可传多次"
    )
    parser.add_argument(
        "--out-img",
        type=str,
        default="memory_decay_curve.png",
        help="输出图文件名"
    )
    args = parser.parse_args()

    plt.figure(figsize=(6,4), dpi=150)

    for spec in args.inputs:
        # 解析 "ModelName=path.csv"
        if "=" not in spec:
            raise ValueError("Each --inputs item must look like ModelName=path.csv")
        model_name, path = spec.split("=", 1)

        turns, curve, label = load_focus_curve(path, model_name)
        plt.plot(turns, curve, marker="o", linewidth=1.5, label=model_name)

    plt.xlabel("Turn index")
    plt.ylabel("Focus correctness (1=on target image)")
    plt.title("Visual Memory Retention over Dialogue Turns")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_img)
    print(f"[saved] {args.out_img}")


if __name__ == "__main__":
    main()
