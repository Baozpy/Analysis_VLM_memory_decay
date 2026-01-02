#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import argparse
from collections import defaultdict

# 将模型名归一化到固定三列：blip2 / llava / qwen
def normalize_model_key(model_str: str):
    s = model_str.lower()
    if "blip2" in s:
        return "blip2"
    if "llava" in s:
        return "llava"
    if "qwen" in s:
        return "qwen"
    return None

# 从文件名解析出 (model_key, sample_id)
# 例如：blip2-opt-6.7b_sample8.csv
FNAME_RE = re.compile(r"(?P<model>[^/\\]+?)_sample(?P<sid>\d+)\.csv$", re.IGNORECASE)

def parse_fname(path: str):
    m = FNAME_RE.search(path)
    if not m:
        return None, None
    model_key = normalize_model_key(m.group("model"))
    if model_key is None:
        return None, None
    sid = int(m.group("sid"))
    return model_key, sid

def read_rows(csv_path: str):
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # turn
            turn_raw = r.get("turn") or r.get("Turn") or r.get("turn_index")
            try:
                turn = int(float(turn_raw)) if turn_raw is not None else None
            except Exception:
                turn = None

            # question: 优先 question，退回 prompt
            question = r.get("question")
            if not question:
                question = r.get("prompt") or ""

            # answer：直接取，不做清洗/偏好选择
            answer = r.get("answer") or r.get("model_answer") or ""

            rows.append({
                "turn": turn,
                "question": question,
                "answer": answer
            })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=str, required=True, help="包含各模型 sampleX.csv 的目录")
    ap.add_argument("--out-csv", type=str, required=True, help="合并后的输出 CSV 路径")
    args = ap.parse_args()

    # 目标表：(sample_id, turn) -> {question, blip2_ans, llava_ans, qwen_ans}
    table = defaultdict(lambda: {"question": "", "blip2_ans": "", "llava_ans": "", "qwen_ans": ""})

    files = [os.path.join(args.input_dir, x) for x in os.listdir(args.input_dir)
             if x.lower().endswith(".csv")]

    for fpath in files:
        model_key, sid = parse_fname(fpath)
        if model_key is None or sid is None:
            continue

        rows = read_rows(fpath)
        for r in rows:
            turn = r["turn"]
            if turn is None:
                # 没有 turn 的行直接跳过（建议上游生成时写入 turn）
                continue

            q = r["question"]
            a = r["answer"]

            key = (sid, turn)
            # 只有在未写入时才赋值（“第一次写入为准”）
            if q and not table[key]["question"]:
                table[key]["question"] = q

            col = f"{model_key}_ans"
            if col in table[key] and not table[key][col]:
                table[key][col] = a

    # 输出
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "turn", "question", "blip2_ans", "llava_ans", "qwen_ans"])
        for (sid, turn) in sorted(table.keys(), key=lambda x: (x[0], x[1])):
            row = table[(sid, turn)]
            writer.writerow([
                sid, turn, row["question"], row["blip2_ans"], row["llava_ans"], row["qwen_ans"]
            ])

    print(f"✅ Done. Merged CSV written to: {args.out_csv}")

if __name__ == "__main__":
    main()
