#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, pandas as pd, numpy as np

ROOT = "real_stats/week3"
Z_FILES = {
    "half_life": os.path.join(ROOT, "Z_compare_half_life.tsv"),
    "mixed_effects": os.path.join(ROOT, "Z_compare_mixed_effects.tsv"),
    "propagation": os.path.join(ROOT, "Z_compare_propagation.tsv"),
}
SIG = 1.96  # 显著性阈值

Z_LONG_CANDIDATES = [
    r"^z$",
    r"^Z$",
    r"z[_-]?score$",
    r"z[_-]?value$",
    r"Zscore$",
    r"Zvalue$",
    r"^t$",
    r"t[_-]?stat$",
]

WIDE_MODEL_PATTERNS = [
    r"^z[_-]?blip2$", r"^z[_-]?llava$", r"^z[_-]?qwen$",
    r"^blip2$", r"^llava$", r"^qwen$",
    r"^blip2[_-]?z$", r"^llava[_-]?z$", r"^qwen[_-]?z$",
]

NUMERIC_EXCLUDE = re.compile(
    r"(?:^p$|pvalue|p_value|^se$|stderr|std[_-]?err|^sd$|^n$|count|mean|avg|diff|delta)",
    re.I
)

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def _ensure_variant(df: pd.DataFrame, path: str) -> pd.DataFrame:
    if "variant" in df.columns:
        return df
    for alt in ["name", "variant_name", "exp", "condition"]:
        if alt in df.columns:
            df = df.rename(columns={alt: "variant"})
            return df
    # 退路：把首列当 variant（打印提示）
    first_col = df.columns[0]
    print(f"[warn] {path} 未找到 variant 列，暂以首列 '{first_col}' 作为 variant")
    df = df.rename(columns={first_col: "variant"})
    return df

def _pick_long_z_col(df: pd.DataFrame):
    for pat in Z_LONG_CANDIDATES:
        for c in df.columns:
            if re.search(pat, c, flags=re.I):
                return c
    return None

def _pick_wide_cols(df: pd.DataFrame):
    found = []
    for pat in WIDE_MODEL_PATTERNS:
        for c in df.columns:
            if re.search(pat, c, flags=re.I):
                found.append(c)
    # 去重、按既定优先顺序挑 blip2/llava/qwen
    pref = ["blip2", "llava", "qwen"]
    normalized = {}
    for c in found:
        key = re.sub(r"^z[_-]?|[_-]?z$", "", c, flags=re.I).lower()  # 去掉前后 z_
        if key in pref and key not in normalized:
            normalized[key] = c
    return [normalized[k] for k in pref if k in normalized]

def _fallback_numeric_as_z(df: pd.DataFrame, path: str):
    num_cols = [c for c in df.columns if c != "variant" and pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in num_cols if not NUMERIC_EXCLUDE.search(c)]
    if not num_cols:
        raise KeyError(f"{path} 找不到可用的 z 列，也没有合适的数值列可退路")
    print(f"[warn] {path} 找不到显式 z 列，将以下数值列作为 z 代理使用：{num_cols}")
    long_rows = []
    for _, r in df.iterrows():
        variant = r["variant"]
        for c in num_cols:
            val = r[c]
            if pd.isna(val): 
                continue
            long_rows.append(dict(variant=variant, model=c.lower(), z=float(val)))
    return pd.DataFrame(long_rows)

def _read_any(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t")
    df = _normalize_cols(df)
    df = _ensure_variant(df, path)

    # 情形A：长表
    zc = _pick_long_z_col(df)
    if zc is not None:
        if zc != "z":
            df = df.rename(columns={zc: "z"})
        # 若无 model 列就不强求
        keep = ["variant", "model", "z"]
        keep = [k for k in keep if k in df.columns]
        return df[keep].copy()

    # 情形B：宽表
    wide = _pick_wide_cols(df)
    if wide:
        long_rows = []
        for _, r in df.iterrows():
            variant = r["variant"]
            for c in wide:
                model = re.sub(r"^z[_-]?|[_-]?z$", "", c, flags=re.I).lower()
                try:
                    val = float(r[c])
                except Exception:
                    val = np.nan
                long_rows.append(dict(variant=variant, model=model, z=val))
        return pd.DataFrame(long_rows)

    # 情形C：退路（任取数值列）
    return _fallback_numeric_as_z(df, path)

def _aggregate(long_df: pd.DataFrame) -> pd.DataFrame:
    g = long_df.groupby("variant")["z"]
    out = g.mean().to_frame("z_mean").reset_index()
    out["z_abs_mean"] = g.apply(lambda s: np.nanmean(np.abs(s))).values
    out["z_max_abs"]  = g.apply(lambda s: np.nanmax(np.abs(s))).values
    return out

def main():
    os.makedirs(ROOT, exist_ok=True)
    rows = []
    summary = ["# Week3 Ablations — Summary", f"- Significance threshold: |Z| ≥ {SIG}", ""]

    votes = {}
    for tag, path in Z_FILES.items():
        df = _read_any(path)
        agg = _aggregate(df)

        top = agg.sort_values("z_mean", ascending=False).iloc[0]
        tname, tz = str(top["variant"]), float(top["z_mean"])
        sig = abs(tz) >= SIG
        votes[tname] = votes.get(tname, 0) + (1 if sig else 0)

        for _, r in agg.iterrows():
            rows.append(dict(
                metric_group=tag,
                variant=r["variant"],
                z_mean=r["z_mean"],
                z_abs_mean=r["z_abs_mean"],
                z_max_abs=r["z_max_abs"],
                sig=abs(r["z_mean"]) >= SIG
            ))

        summary += [
            f"## {tag}",
            f"- Best: **{tname}**  (Z̄ = {tz:.2f}, {'significant' if sig else 'ns'})",
            f"- Top-5 by Z̄:",
        ]
        for _, r in agg.sort_values("z_mean", ascending=False).head(5).iterrows():
            summary.append(f"  - {r['variant']}: Z̄={r['z_mean']:.2f} (|Z|̄={r['z_abs_mean']:.2f}, |Z|max={r['z_max_abs']:.2f})")
        summary.append("")

    if votes:
        overall = sorted(votes.items(), key=lambda x: x[1], reverse=True)[0]
        summary += ["## Overall pick", f"- **{overall[0]}** (votes = {overall[1]})", ""]

    out_df = pd.DataFrame(rows).sort_values(["metric_group","z_mean"], ascending=False)
    out_tsv = os.path.join(ROOT, "week3_all_z.tsv")
    out_md  = os.path.join(ROOT, "week3_summary.md")
    out_df.to_csv(out_tsv, sep="\t", index=False)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    print("[OK] wrote", out_tsv)
    print("[OK] wrote", out_md)

if __name__ == "__main__":
    main()
