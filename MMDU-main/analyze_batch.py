#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, argparse
import pandas as pd
from collections import defaultdict

# -----------------------
# 文件名解析：<model>_sample<id>.csv
# 例如：blip2-opt-6.7b_sample8.csv
# -----------------------
FNAME_RE = re.compile(r"([^/\\]+)_sample(\d+)\.csv$", re.IGNORECASE)

def normalize_model_key(s: str) -> str:
    s = s.lower()
    if "blip2" in s: return "blip2"
    if "llava" in s: return "llava"
    if "qwen"  in s: return "qwen"
    return s

def list_csvs(root: str):
    return [os.path.join(root, x) for x in os.listdir(root) if x.lower().endswith(".csv")]

def parse_fname(path: str):
    m = FNAME_RE.search(path)
    if not m:
        return None, None
    return normalize_model_key(m.group(1)), int(m.group(2))

# -----------------------
# “温和阈值”文本质量判定
# -----------------------
import re as _re

KEYWORDS = {
    "dome","arch","arches","fountain","trevi","statue","sarcophagus",
    "brick","roof","facade","façade","column","columns","gazebo",
    "temple","church","cathedral","tomb","crypt","mausoleum"
}
_SYMBOLIC_SHORTS = {".","...","n/a","null","none"}

def _num_tokens_fast(text: str) -> int:
    return len(_re.findall(r"\w+|[^\w\s]", text or ""))

def _has_keywords(text: str) -> bool:
    if not text: return False
    low = text.lower()
    return any(k in low for k in KEYWORDS)

def is_answer_ok_mild(answer: str,
                      truncation_hit: int = 0,
                      hallucination_level: int = 0,
                      min_tokens: int = 8) -> bool:
    """
    温和阈值：
      - 过滤纯符号/过短答
      - 严重幻觉(>=2)判为不 OK；轻微(<=1)不过度惩罚
      - 若被截断但长度足够(>=40)或包含关键实体词，也可判 OK
    """
    a = (answer or "").strip()
    if not a or a in _SYMBOLIC_SHORTS:
        return False
    ntok = _num_tokens_fast(a)
    if ntok < min_tokens:
        return False
    try:
        hl = int(hallucination_level)
    except Exception:
        hl = 0
    if hl >= 2:
        return False
    try:
        th = int(truncation_hit)
    except Exception:
        th = 0
    if th:
        return (ntok >= 40) or _has_keywords(a)
    return True

# -----------------------
# 读入单个 CSV（容错）
# -----------------------
def _read_csv_robust(path: str) -> pd.DataFrame:
    # 先尝试常规读取
    try:
        return pd.read_csv(path)
    except Exception:
        pass
    # 再尝试 on_bad_lines='skip'
    try:
        return pd.read_csv(path, on_bad_lines='skip', engine='python')
    except Exception:
        pass
    # 再尝试分隔符容错
    try:
        return pd.read_csv(path, sep=None, engine='python', on_bad_lines='skip')
    except Exception as e:
        raise e

def load_dialogue_csv(path: str) -> pd.DataFrame:
    df = _read_csv_robust(path)
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # 统一列名（不区分大小写）
    cols_lower = {c.lower(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in cols_lower: return cols_lower[c]
        return None

    turn_c = pick("turn","turn_index")
    q_c    = pick("question","prompt")
    a_c    = pick("answer","model_answer","response")
    tgt_c  = pick("target_img","target","target_image")
    pred_c = pick("predicted_img","pred","pred_image")
    foc_c  = pick("focus_correct","focus","on_target")
    hall_c = pick("hallucination_level","hallucination","hall_level")
    trunc_c= pick("truncation_hit","truncated","hit_trunc")

    out = pd.DataFrame()
    if turn_c: out["turn"] = pd.to_numeric(df[turn_c], errors="coerce").astype("Int64")
    if q_c:    out["question"] = df[q_c].astype(str)
    if a_c:    out["answer"] = df[a_c].astype(str)
    if tgt_c:  out["target_img"] = pd.to_numeric(df[tgt_c], errors="coerce")
    if pred_c: out["predicted_img"] = pd.to_numeric(df[pred_c], errors="coerce")
    if foc_c:  out["focus_correct"] = pd.to_numeric(df[foc_c], errors="coerce")
    if hall_c: out["hallucination_level"] = pd.to_numeric(df[hall_c], errors="coerce")
    if trunc_c:out["truncation_hit"] = pd.to_numeric(df[trunc_c], errors="coerce")
    return out

# -----------------------
# 单文件聚合为 turn 粒度
# -----------------------
def analyze_one(df: pd.DataFrame, exclude_missing_pred: bool) -> pd.DataFrame:
    if df is None or df.empty: 
        return pd.DataFrame(columns=["turn","focus_rate","focus_n","answer_ok_rate","answer_ok_n","trunc_rate","trunc_n"])

    df = df.dropna(subset=["turn"]).copy()

    # 视觉聚焦：优先用 focus_correct；否则回退 target==pred；再没有则 NA
    if "focus_correct" not in df.columns:
        if "target_img" in df.columns and "predicted_img" in df.columns:
            df["focus_correct"] = (df["target_img"] == df["predicted_img"]).astype("float")
        else:
            df["focus_correct"] = pd.NA

    # 文本质量：温和阈值
    if "answer" not in df.columns:
        df["answer"] = ""
    if "truncation_hit" not in df.columns:
        df["truncation_hit"] = 0
    if "hallucination_level" not in df.columns:
        df["hallucination_level"] = 0

    df["answer_ok"] = [
        int(is_answer_ok_mild(a, t, h))
        for a, t, h in zip(df["answer"], df["truncation_hit"], df["hallucination_level"])
    ]

    # 计算聚焦时是否忽略缺失 predicted_img
    if exclude_missing_pred and "predicted_img" in df.columns:
        focus_base = df[~df["predicted_img"].isna()].copy()
    else:
        focus_base = df.copy()

    # 逐 turn 聚合
    rows = []
    for t, g in df.groupby("turn"):
        row = {"turn": int(t)}

        # 聚焦率
        g_focus = focus_base[focus_base["turn"] == t]
        if len(g_focus) > 0 and "focus_correct" in g_focus.columns:
            fc = g_focus["focus_correct"].dropna()
            row["focus_rate"] = float(fc.mean()) if len(fc) > 0 else float("nan")
            row["focus_n"] = int(len(fc))
        else:
            row["focus_rate"] = float("nan")
            row["focus_n"] = 0

        # 文本 OK 率
        row["answer_ok_rate"] = float(g["answer_ok"].mean()) if len(g) > 0 else float("nan")
        row["answer_ok_n"] = int(len(g))

        # 截断率
        if "truncation_hit" in g.columns:
            th = pd.to_numeric(g["truncation_hit"], errors="coerce").fillna(0)
            row["trunc_rate"] = float(th.mean())
            row["trunc_n"] = int(len(th))
        else:
            row["trunc_rate"] = float("nan")
            row["trunc_n"] = 0

        rows.append(row)

    return pd.DataFrame(rows).sort_values("turn")

# -----------------------
# 主程序：批量处理目录中所有 *_sampleX.csv
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="目录：含各模型的 *_sampleX.csv")
    ap.add_argument("--aggregate-out", required=True, help="输出：全局聚合 CSV")
    ap.add_argument("--exclude-missing-pred", action="store_true",
                    help="计算聚焦率时忽略没有 predicted_img 的行")
    args = ap.parse_args()

    files = list_csvs(args.input_dir)
    buckets = defaultdict(dict)  # (model, sample_id) -> path
    for p in files:
        mk, sid = parse_fname(p)
        if mk is None or sid is None:
            print(f"[warn] skip (name not match): {p}")
            continue
        buckets[(mk, sid)]["path"] = p

    os.makedirs(args.input_dir, exist_ok=True)

    agg_all = []
    for (mk, sid), info in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        p = info["path"]
        try:
            df_raw = load_dialogue_csv(p)
            if df_raw.empty:
                print(f"[warn] empty csv: {p}")
                continue

            df_an = analyze_one(df_raw, exclude_missing_pred=args.exclude_missing_pred)
            df_an["model"] = mk
            df_an["sample_id"] = sid

            # per-sample 输出
            out_name = f"{mk}_sample{sid}_analysis.csv"
            out_path = os.path.join(args.input_dir, out_name)
            df_an.to_csv(out_path, index=False, encoding="utf-8")

            agg_all.append(df_an)
        except Exception as e:
            print(f"[warn] skip {p}: {e}")

    if len(agg_all) == 0:
        # 空文件也给表头，避免后续画图脚本崩溃
        empty = pd.DataFrame(columns=[
            "model","sample_id","turn","focus_rate","focus_n",
            "answer_ok_rate","answer_ok_n","trunc_rate","trunc_n"
        ])
        empty.to_csv(args.aggregate_out, index=False, encoding="utf-8")
        print(f"[analyze_batch] no valid CSVs found. wrote empty {args.aggregate_out}")
        return

    agg_df = pd.concat(agg_all, ignore_index=True)
    agg_df = agg_df[[
        "model","sample_id","turn",
        "focus_rate","focus_n",
        "answer_ok_rate","answer_ok_n",
        "trunc_rate","trunc_n"
    ]]
    agg_df.to_csv(args.aggregate_out, index=False, encoding="utf-8")
    print(f"[analyze_batch] wrote per-sample *_analysis.csv and {args.aggregate_out}")

if __name__ == "__main__":
    main()
