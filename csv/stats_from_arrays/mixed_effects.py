# mixed_effects.py
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

def fit_gee(df, xcols, ycol="ok", cluster="sample_id", cov_struct="exchangeable"):
    df = df.dropna(subset=xcols+[ycol, cluster]).copy()
    df["intercept"] = 1.0
    X = df[["intercept"] + xcols]
    y = df[ycol].astype(int)
    groups = df[cluster].astype(str)
    cov = sm.cov_struct.Exchangeable() if cov_struct=="exchangeable" else sm.cov_struct.Independence()
    fam = sm.families.Binomial()
    model = sm.GEE(y, X, groups=groups, cov_struct=cov, family=fam)
    res = model.fit()
    out = res.summary().as_text()
    return res, out

def _stat_values(res):
    # 兼容不同 statsmodels 版本的命名
    if hasattr(res, "z_values"):    # 常见于 GEE
        return getattr(res, "z_values")
    if hasattr(res, "zvalues"):     # 老版本
        return getattr(res, "zvalues")
    if hasattr(res, "tvalues"):     # 回退用 t 统计量
        return getattr(res, "tvalues")
    # 最后兜底：z ≈ coef / se
    return res.params / res.bse

def _conf_int_safe(res, alpha=0.05):
    try:
        ci = res.conf_int(alpha=alpha)
        # 有的版本返回 DataFrame，有的返回 ndarray，这里都兼容
        if hasattr(ci, "iloc"):
            lo = ci.iloc[:, 0].values
            hi = ci.iloc[:, 1].values
        else:
            lo = ci[:, 0]
            hi = ci[:, 1]
    except Exception:
        # 若没有 conf_int，则用正态近似：coef ± 1.96*se
        q = 1.96
        lo = res.params.values - q * res.bse.values
        hi = res.params.values + q * res.bse.values
    return lo, hi

def one_table(res, model_name, xcols=None):
    z_or_t = _stat_values(res)
    # pvalues 兼容
    if hasattr(res, "pvalues"):
        pvals = res.pvalues.values if hasattr(res.pvalues, "values") else res.pvalues
    else:
        # 极端兜底：无 p 值时给 NA
        import numpy as np
        pvals = np.full(len(res.params), np.nan)

    ci_lo, ci_hi = _conf_int_safe(res)

    import pandas as pd
    df = pd.DataFrame({
        "term":   res.params.index.astype(str),
        "coef":   res.params.values,
        "std_err":res.bse.values,
        "z":      z_or_t.values if hasattr(z_or_t, "values") else z_or_t,
        "p":      pvals,
        "ci_lo":  ci_lo,
        "ci_hi":  ci_hi,
    })

    # 过滤截距
    df = df[~df["term"].str.lower().isin(["intercept", "const"])].copy()

    # 效果名映射（保留你原来的展示）
    effect_map = {
        "delta": "Turns since first mention (Δ)",
        "k":     "Revisit index (k)",
        "tau":   "Normalized progress (τ)",
    }
    df.insert(0, "vlm", model_name)
    df["effect"] = df["term"].map(effect_map).fillna(df["term"])

    # 保持原脚本期望的列顺序
    return df[["vlm", "effect", "coef", "ci_lo", "ci_hi", "p"]]


def run_for(path, xcol, label):
    df = pd.read_csv(path, sep="\t")
    out = []
    for m in df["model"].unique():
        sub = df[df.model==m].copy()
        res, _ = fit_gee(sub, [xcol])
        out.append(one_table(res, m, [xcol]))
    return pd.concat(out).assign(metric=label)

if __name__ == "__main__":
    pΔ = Path("stats_Δ.tsv")
    pk = Path("stats_k.tsv")
    pτ = Path("stats_τ.tsv")

    tabΔ = run_for(pΔ, "delta", "B2")
    tabk = run_for(pk, "k", "D")
    tabτ = run_for(pτ, "tau", "B1")

    alltab = pd.concat([tabΔ, tabk, tabτ], ignore_index=True)
    alltab.to_csv("gee_effects_summary.csv", index=False)

    # 生成一个可直接贴到 slides 的粗体 markdown 表
    def fmt(r):
        star = "" if r.p>0.05 else ("*" if r.p>0.01 else "**")
        return f"| {r.metric} | {r.vlm} | {r.effect} | {r.coef:.3f} [{r.ci_lo:.3f},{r.ci_hi:.3f}] | p={r.p:.3g}{star} |"
    lines = ["| Metric | VLM | Effect | Coef [95% CI] | p-value |","|---|---|---|---|---|"]
    for _,r in alltab.iterrows():
        lines.append(fmt(r))
    Path("gee_effects_pretty.md").write_text("\n".join(lines), encoding="utf-8")
    print("Saved: gee_effects_summary.csv, gee_effects_pretty.md")
