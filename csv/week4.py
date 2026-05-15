python - <<'PY'
import os, math, numpy as np, pandas as pd
from pathlib import Path
import subprocess, sys

# ========= 可改参数 =========
WEEK = "week5"
KS   = [1,2,3,4,6,8,12,16]   # 需要跑的 k 集合
# ===========================

ROOT = Path(".").resolve()
ARR_DIR = ROOT / f"{WEEK}_ablations" / "variants_out"
OUT_DIR = ROOT / "real_stats" / WEEK
BUILD   = ROOT / "stats_from_arrays" / "build_stats_from_samples.py"
MIXED   = ROOT / "stats_from_arrays" / "mixed_effects.py"

OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_tsv(p):
    p = Path(p)
    if not p.exists(): return pd.DataFrame()
    try: return pd.read_csv(p, sep="\t")
    except Exception: return pd.DataFrame()

def refit_half_life_from_events(ev_path: Path, out_path: Path, lam_eps=1e-3):
    """从 events.tsv 拟合 half_life.tsv，避免旧 Δ 表的依赖。"""
    df = read_tsv(ev_path)
    need = {"model","turn","ok"}
    if df.empty or not need.issubset(df.columns):
        return False
    g = (df.rename(columns={"turn":"delta"})
           .groupby(["model","delta"])
           .agg(ok_mean=("ok","mean"), n=("ok","size"))
           .reset_index()
           .sort_values(["model","delta"]))
    rows = []
    for m, sub in g.groupby("model"):
        x = sub["delta"].astype(float).to_numpy()
        y = np.clip(sub["ok_mean"].astype(float).to_numpy(), 1e-6, 1.0)
        w = sub["n"].astype(float).to_numpy()
        if x.size < 2 or np.isclose(x.var(), 0):
            lam = np.nan; A = float(y.mean()); t_half = np.nan
        else:
            Y = np.log(y)
            try:
                slope, intercept = np.polyfit(x, Y, 1, w=w)
                lam = float(-slope)
                if not np.isfinite(lam) or lam <= lam_eps:
                    lam = np.nan; t_half = np.nan
                else:
                    t_half = float(math.log(2)/lam)
                A = float(np.exp(intercept))
            except Exception:
                lam = np.nan; A = float(np.exp(np.mean(Y))); t_half = np.nan
        rows.append(dict(model=m, points=int(sub["n"].sum()),
                         delta_min=float(x.min()) if x.size else np.nan,
                         delta_max=float(x.max()) if x.size else np.nan,
                         A=A, lambda_=lam, t_half=t_half))
    hf = pd.DataFrame(rows, columns=["model","points","delta_min","delta_max","A","lambda_","t_half"])
    hf.to_csv(out_path, sep="\t", index=False)
    return True

def z_from_mixed(df):
    if df.empty: return None
    sub = df[df.get("xvar","B2").eq("B2")] if "xvar" in df.columns else df
    sub = sub[sub.get("term","term").ne("const")]
    if sub.empty or "coef" not in sub.columns or "se" not in sub.columns:
        return None
    coef, se = float(sub.iloc[0]["coef"]), float(sub.iloc[0]["se"])
    if not np.isfinite(se) or se==0: return None
    return coef/se

# ========== Step 1：逐 k 构建 raw，拟合 half-life，跑 mixed-effects ==========
for k in KS:
    arr = ARR_DIR / f"recap_k{k}_fixed.py"
    out = OUT_DIR / f"recap_k{k}"
    out.mkdir(parents=True, exist_ok=True)
    if not arr.exists():
        print(f"[skip] {arr} 不存在，跳过 k={k}")
        continue
    print(f"== [k={k}] build_stats_from_samples ==")
    subprocess.check_call([sys.executable, str(BUILD), "--arrays", str(arr), "--out_dir", str(out)])

    ev = out / "events.tsv"
    print(f"== [k={k}] fit half_life (from events) ==")
    ok_hf = refit_half_life_from_events(ev, out/"half_life.tsv")
    print(f"[half-life] k={k}: {'OK' if ok_hf else 'SKIP'} -> {out/'half_life.tsv'}")

    print(f"== [k={k}] mixed_effects ==")
    subprocess.check_call([sys.executable, str(MIXED), "--in-dir", str(out), "--out-dir", str(out)])

# ========== Step 2：汇总为 all_z / summary_table / rank / md ==========
rows_all, rows_tbl = [], []

def push_metric(variant, metric_group, zvals):
    zvals = [z for z in zvals if z is not None and np.isfinite(z)]
    if not zvals: return
    z_mean = float(np.mean(zvals))
    rows_all.append(dict(metric_group=metric_group, variant=variant, z=z_mean))
    rows_tbl.append(dict(metric_group=metric_group, variant=variant,
                         z_mean=z_mean,
                         z_abs_mean=float(np.mean(np.abs(zvals))),
                         z_max_abs=float(np.max(np.abs(zvals))),
                         sig=(abs(z_mean) >= 1.96) if metric_group=="mixed_effects" else ""))

# half_life：取各模型 t½ 的中位数
for k in KS:
    hf = read_tsv(OUT_DIR/f"recap_k{k}"/"half_life.tsv")
    if not hf.empty and "t_half" in hf.columns:
        vals = pd.to_numeric(hf["t_half"], errors="coerce")
        vals = vals[np.isfinite(vals)]
        vals = vals[vals > 0]
        if not vals.empty:
            push_metric(f"recap_k{k}", "half_life", [float(np.median(vals))])

# mixed_effects：取 B2 斜率的 z
for k in KS:
    mx = read_tsv(OUT_DIR/f"recap_k{k}"/"mixed_effects.tsv")
    z = z_from_mixed(mx)
    if z is not None:
        push_metric(f"recap_k{k}", "mixed_effects", [z])

all_z = pd.DataFrame(rows_all, columns=["metric_group","variant","z"]).sort_values(["metric_group","z"])
tbl   = pd.DataFrame(rows_tbl, columns=["metric_group","variant","z_mean","z_abs_mean","z_max_abs","sig"])

all_z.to_csv(OUT_DIR/"week5_all_z.tsv", sep="\t", index=False)
tbl.to_csv(OUT_DIR/"week5_summary_table.tsv", sep="\t", index=False)
print("[OK] wrote", OUT_DIR/"week5_all_z.tsv", "rows:", len(all_z))
print("[OK] wrote", OUT_DIR/"week5_summary_table.tsv", "rows:", len(tbl))

# 排名：mixed_effects 越小越好；half_life 越大越好
rank_rows = []
for metric in ["mixed_effects","half_life"]:
    sub = all_z[all_z["metric_group"]==metric].copy()
    if sub.empty: continue
    if metric == "mixed_effects":
        sub = sub.assign(z_for_rank=sub["z"].abs()).sort_values("z_for_rank", ascending=True)
    else:
        sub = sub.assign(z_for_rank=sub["z"]).sort_values("z_for_rank", ascending=False)
    sub["rank"] = np.arange(1, len(sub)+1, dtype=int)

    sub["rank"] = np.arange(1, len(sub)+1, dtype=int)
    rank_rows.append(sub[["variant","rank"]])
if rank_rows:
    r = pd.concat(rank_rows, ignore_index=True)
    rank = (r.groupby("variant")["rank"]
              .agg(rank_mean="mean", rank_std="std", rank_min="min", rank_max="max", n_metrics="count")
              .reset_index()
              .sort_values("rank_mean"))
else:
    rank = pd.DataFrame(columns=["variant","rank_mean","rank_std","rank_min","rank_max","n_metrics"])
rank.to_csv(OUT_DIR/"rank_summary.tsv", sep="\t", index=False)
print("[OK] wrote", OUT_DIR/"rank_summary.tsv")

# MD 摘要
lines = ["# Week5 Ablations — Summary", "- Significance threshold: |Z| ≥ 1.96", ""]
for metric in ["half_life","mixed_effects"]:
    sub = all_z[all_z["metric_group"]==metric]
    if sub.empty: continue
    if metric=="mixed_effects":
        best = sub.sort_values("z", ascending=True).iloc[0]
        lines += ["", "## mixed_effects",
                  f"- Best: **{best['variant']}**  (Z̄ = {best['z']:.2f}{'' if abs(best['z'])>=1.96 else ', ns'})",
                  "- Top-5 by Z̄:"]
        for _, r in sub.sort_values("z", ascending=True).head(5).iterrows():
            lines.append(f"- {r['variant']}: Z̄={r['z']:.2f}{' — significant' if abs(r['z'])>=1.96 else ''}")
    else:
        best = sub.sort_values("z", ascending=False).iloc[0]
        lines += ["", "## half_life",
                  f"- Best: **{best['variant']}**  (t½ ≈ {best['z']:.2f})",
                  "- Top-5 by t½:"]
        for _, r in sub.sort_values("z", ascending=False).head(5).iterrows():
            lines.append(f"- {r['variant']}: t½≈{r['z']:.2f}")
lines += ["", "## Overall pick"]
if not rank.empty:
    top = rank.iloc[0]
    lines.append(f"- **{top['variant']}** (rank_mean = {float(top['rank_mean']):.2f})")
else:
    lines.append("- _(no variants)_")
lines += ["", "", "## Rank table (lower is better)",
          "variant\trank_mean\trank_std\trank_min\trank_max\tn_metrics"]
for _, r in rank.iterrows():
    lines.append(f"{r['variant']}\t{float(r['rank_mean']):.2f}\t{0.0 if pd.isna(r['rank_std']) else float(r['rank_std']):.2f}\t{float(r['rank_min']):.1f}\t{float(r['rank_max']):.1f}\t{int(r['n_metrics'])}")
(OUT_DIR/"week5_summary.md").write_text("\n".join(lines), encoding="utf-8")
print("[OK] wrote", OUT_DIR/"week5_summary.md")
PY
