#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze VLM pilot CSV results:
- Computes EchoRate, 3-gram repetition, answer length, echo flags, truncation, etc.
- Aggregates per-turn trends and prints a concise summary.
- Exports per-row metrics (analysis_metrics.csv) and summary (analysis_summary.txt).

Usage:
  python analyze_results.py pilot_results.csv
"""

import sys, csv, re, math, statistics
from collections import Counter, defaultdict
from pathlib import Path

PUNCT_END = (".", "!", "?", "。", "！", "？")
WEIRD_MARKERS = ("[NEXT]", "[IMAGE]", "<IMAGEHERE>", "[PREV]", "PREVIOUSLY ASKED")

def tokenize(s: str):
    return re.findall(r"\w+", (s or "").lower())

def ngrams(seq, n=3):
    return [' '.join(seq[i:i+n]) for i in range(len(seq)-n+1)]

def echo_rate(prompt: str, answer: str) -> float:
    a = tokenize(answer)
    if not a: return 0.0
    pset = set(tokenize(prompt))
    overlap = sum(1 for w in a if w in pset)
    return overlap / len(a)

def rep_rate(answer: str, n=3) -> float:
    a = tokenize(answer)
    gs = ngrams(a, n)
    if not gs: return 0.0
    cnt = Counter(gs)
    rep = sum(c for g, c in cnt.items() if c > 1)
    return rep / len(gs)

def ans_len(answer: str) -> int:
    return len(tokenize(answer))

def looks_truncated(answer: str) -> bool:
    """简单启发式：无正常结尾标点 & 最后一个词看起来被截断（如以逗号、and、空结尾）"""
    if not answer: return False
    tail = answer.strip()[-1]
    if tail in PUNCT_END:
        return False
    # 最后 1~2 个 token 是否是开放连接词或明显未完
    tail_tokens = tokenize(answer)[-2:]
    if not tail_tokens: 
        return True
    if tail_tokens[-1] in {"and", "or", "but", "because"}:
        return True
    # 过短但没结束标点
    if len(answer.strip()) < 20:
        return False
    return True

def has_weird_markers(answer: str) -> bool:
    s = (answer or "").upper()
    return any(m in s for m in WEIRD_MARKERS)

def is_echo_strict(prompt: str, answer: str) -> bool:
    """严格复读：完全相同或回答大部分包含 prompt 文本"""
    p = (prompt or "").strip()
    a = (answer or "").strip()
    if not a: return False
    if a == p: 
        return True
    # 长度相近且 answer 含有大段 prompt
    if len(a) >= 0.9 * len(p) and p[: max(10, len(p)//3)] in a:
        return True
    return False

def safe_float(v):
    try:
        return float(v)
    except:
        return float("nan")

def main():
    in_csv = Path(sys.argv[1] if len(sys.argv) > 1 else "pilot_results.csv")
    if not in_csv.exists():
        print(f"[Error] CSV not found: {in_csv}")
        sys.exit(1)

    rows = []
    with in_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        # 兼容大小写/不同列名
        field_map = {k.lower(): k for k in r.fieldnames}
        def col(name): return field_map.get(name.lower(), name)
        for row in r:
            rows.append({
                "sample_id": row.get(col("sample_id"), row.get(col("id"), "")),
                "turn": row.get(col("turn"), ""),
                "img_idx": row.get(col("img_idx"), ""),
                "prompt": row.get(col("prompt"), ""),
                "answer": row.get(col("answer"), "")
            })

    metrics = []
    by_turn = defaultdict(list)

    for row in rows:
        prompt = row["prompt"]
        answer = row["answer"]
        turn = int(row["turn"]) if str(row["turn"]).isdigit() else None

        e = echo_rate(prompt, answer)
        rep = rep_rate(answer, 3)
        length = ans_len(answer)
        echo_flag = is_echo_strict(prompt, answer)
        trunc = looks_truncated(answer)
        weird = has_weird_markers(answer)

        met = {
            "sample_id": row["sample_id"],
            "turn": turn,
            "img_idx": row["img_idx"],
            "EchoRate": round(e, 4),
            "Rep3g": round(rep, 4),
            "AnsLen": length,
            "IsEcho": int(echo_flag),
            "MaybeTruncated": int(trunc),
            "HasWeirdMarkers": int(weird),
        }
        metrics.append(met)
        if turn is not None:
            by_turn[turn].append(met)

    # 导出逐行指标
    out_detail = in_csv.with_name("analysis_metrics.csv")
    with out_detail.open("w", newline="", encoding="utf-8") as g:
        w = csv.writer(g)
        w.writerow(["sample_id","turn","img_idx","EchoRate","Rep3g","AnsLen","IsEcho","MaybeTruncated","HasWeirdMarkers"])
        for m in metrics:
            w.writerow([m["sample_id"], m["turn"], m["img_idx"], m["EchoRate"], m["Rep3g"], m["AnsLen"], m["IsEcho"], m["MaybeTruncated"], m["HasWeirdMarkers"]])

    # 汇总
    def avg(xs): 
        xs = [x for x in xs if not math.isnan(x)]
        return (sum(xs)/len(xs)) if xs else float("nan")
    def med(xs):
        xs = [x for x in xs if not math.isnan(x)]
        return statistics.median(xs) if xs else float("nan")

    echo_all = [safe_float(m["EchoRate"]) for m in metrics]
    rep_all  = [safe_float(m["Rep3g"]) for m in metrics]
    len_all  = [safe_float(m["AnsLen"]) for m in metrics]
    echo_pct = 100.0 * sum(m["IsEcho"] for m in metrics) / len(metrics) if metrics else 0.0
    trunc_pct= 100.0 * sum(m["MaybeTruncated"] for m in metrics) / len(metrics) if metrics else 0.0
    weird_pct= 100.0 * sum(m["HasWeirdMarkers"] for m in metrics) / len(metrics) if metrics else 0.0

    lines = []
    lines.append(f"Input CSV: {in_csv.name}")
    lines.append(f"Total rows: {len(metrics)}")
    lines.append("")
    lines.append("== Overall ==")
    lines.append(f"EchoRate    avg={avg(echo_all):.3f}  med={med(echo_all):.3f}")
    lines.append(f"Rep3g       avg={avg(rep_all):.3f}   med={med(rep_all):.3f}")
    lines.append(f"AnsLen      avg={avg(len_all):.1f}   med={med(len_all):.1f}")
    lines.append(f"IsEcho      {echo_pct:.1f}%")
    lines.append(f"Truncated   {trunc_pct:.1f}%")
    lines.append(f"WeirdMarker {weird_pct:.1f}%")
    lines.append("")

    turns_sorted = sorted(by_turn.keys())
    if turns_sorted:
        lines.append("== By Turn ==")
        lines.append("turn\tN\tEchoRate_avg\tRep3g_avg\tAnsLen_avg\tIsEcho% \tTrunc%")
        for t in turns_sorted:
            arr = by_turn[t]
            er = avg([m["EchoRate"] for m in arr])
            rr = avg([m["Rep3g"] for m in arr])
            la = avg([m["AnsLen"] for m in arr])
            ie = 100.0 * sum(m["IsEcho"] for m in arr) / len(arr)
            tr = 100.0 * sum(m["MaybeTruncated"] for m in arr) / len(arr)
            lines.append(f"{t}\t{len(arr)}\t{er:.3f}\t\t{rr:.3f}\t\t{la:.1f}\t\t{ie:.1f}%\t{tr:.1f}%")

    summary_txt = in_csv.with_name("analysis_summary.txt")
    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\n✅ Saved: {out_detail.name}, {summary_txt.name}")

if __name__ == "__main__":
    main()
