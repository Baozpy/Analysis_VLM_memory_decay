# week3_ablations/rank_robust.py
import pandas as pd
from pathlib import Path

p = Path("real_stats/week3/week3_all_z.tsv")
df = pd.read_csv(p, sep="\t")

# 我们只用聚合后的表（metric_group, variant, z_mean）
core = df.dropna(subset=["metric_group","variant","z_mean"])[["metric_group","variant","z_mean"]].copy()

def rank_one(g):
    # 越大越好 → rank 1 是最优
    g = g.sort_values("z_mean", ascending=False).reset_index(drop=True)
    g["rank"] = g.index + 1
    return g

ranked = core.groupby("metric_group", group_keys=False).apply(rank_one)

# 总结每个 variant 的平均名次（跨 metric_group）
summary = (
    ranked.groupby("variant")["rank"]
    .agg(["mean","std","min","max","count"])
    .sort_values("mean")
    .reset_index()
    .rename(columns={"mean":"rank_mean","std":"rank_std","min":"rank_min","max":"rank_max","count":"n_metrics"})
)

out1 = Path("real_stats/week3/week3_rank_summary.tsv")
summary.to_csv(out1, sep="\t", index=False)

# 如需分模型一致性（如果你的 week3_all_z.tsv 含 model 列）：
if "model" in df.columns:
    per_model = (
        df.dropna(subset=["model","metric_group","variant","z_mean"])
          .groupby(["model","metric_group","variant"])["z_mean"].mean().reset_index()
    )
    per_model_ranked = (
        per_model.groupby(["model","metric_group"], group_keys=False)
                 .apply(rank_one)
    )
    per_model_summary = (
        per_model_ranked.groupby(["model","variant"])["rank"]
                        .agg(["mean","std","min","max","count"])
                        .reset_index()
                        .sort_values(["model","mean"])
                        .rename(columns={"mean":"rank_mean","std":"rank_std","min":"rank_min","max":"rank_max","count":"n_metrics"})
    )
    per_model_out = Path("real_stats/week3/week3_rank_per_model.tsv")
    per_model_summary.to_csv(per_model_out, sep="\t", index=False)

print(f"[OK] wrote {out1}")
