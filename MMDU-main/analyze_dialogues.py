# analyze_dialogues.py
import re
import argparse
import pandas as pd
from typing import Dict, List, Tuple, Optional

###############################################################################
# 1. 视觉签名词典（示例：sample8）
#
# 目的：
#   - 每张图有哪些专属的视觉关键词
#   - 用于判断模型回答“在说哪张图”
#
# 用法：
#   - 修改这些 list 以适配你的样本
#   - key 是图的编号（1-based，对应 Image1, Image2,...）
###############################################################################

IMAGE_SIGNATURES_SAMPLE8 = {
    1: [
        "white dome", "dome", "temple", "house of worship",
        "bahá", "bahai", "bahai house", "lotus", "religious",
        "marble steps", "many windows", "shrine"
    ],
    2: [
        "fountain", "trevi", "rome", "roman", "coins",
        "mythological", "goddess", "sculpture", "archway",
        "baroque facade", "water cascading"
    ],
    3: [
        "sarcophagus", "tomb", "crypt", "casket",
        "napoleon", "emperor", "circular hall", "marble floor",
        "rotunda", "honor", "resting place"
    ],
    4: [
        "red brick", "ivy", "garden", "gazebo",
        "tower", "clock tower", "turret", "conical roof",
        "courtyard", "brick exterior"
    ],
}


###############################################################################
# 2. 幻觉级别粗判（启发式）
#    level = 2: 重度编故事/编历史（强烈的虚构 narrative）
#    level = 1: 轻度泛化/模糊社会故事
#    level = 0: 基本客观描述
###############################################################################

STRONG_HALLUC_PHRASES = [
    # 明显虚构身份/历史叙事，和图像本身不一定匹配
    "the founder of", "he was the first president", "george washington",
    "they built this to show how much they loved",
    "this symbolizes how much they loved him",
    "he was born", "his followers", "worshippers wanted to show",
    "center of the city of", "tourist attraction"  # often BS in BLIP-2 answers
]

MILD_HALLUC_PHRASES = [
    "people gather here to celebrate", "it represents unity",
    "it is used to symbolize inclusivity", "it shows harmony of all faiths",
    "this is about peace for all religions",
    "warmth and comfort", "inviting visitors to explore",
    "communal space for gathering and timekeeping",
    "a narrative of unity and harmony"
]


def get_hallucination_level(answer: str) -> int:
    """返回 0/1/2"""
    low = answer.lower()
    for kw in STRONG_HALLUC_PHRASES:
        if kw in low:
            return 2
    for kw in MILD_HALLUC_PHRASES:
        if kw in low:
            return 1
    return 0


###############################################################################
# 3. 从 prompt 解析出“问题文本”和“目标图编号”
###############################################################################

def extract_question_from_prompt(prompt: str) -> str:
    """
    你的 prompt 是这种结构：
    "You are given ... Q: (真实问题) A:"
    我们想取 Q: 和 A: 之间那块当作原始问题（更干净）。
    """
    # Try "Q:" ... "A:"
    m = re.search(r"Q:\s*(.*?)\s*A:\s*$", prompt, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    # fallback: last line
    lines = prompt.strip().splitlines()
    return lines[-1].strip() if lines else prompt.strip()


def extract_target_image_id(question: str) -> Optional[int]:
    """
    找到最后一次出现的 "ImageK" 并返回 K (int)
    如果没有出现，就返回 None
    """
    matches = re.findall(r"image\s*(\d+)", question, flags=re.I)
    if not matches:
        return None
    last_k = matches[-1]
    try:
        return int(last_k)
    except ValueError:
        return None


###############################################################################
# 4. 基于回答推测“模型实际上在描述哪张图”
#    算法：对每个 image_i 的签名词做计数，选命中最多的
###############################################################################

def infer_image_from_answer(
    answer: str,
    image_signatures: Dict[int, List[str]]
) -> Optional[int]:
    low = answer.lower()
    best_img = None
    best_score = 0
    for img_id, keywords in image_signatures.items():
        score = 0
        for kw in keywords:
            if kw.lower() in low:
                score += 1
        if score > best_score:
            best_score = score
            best_img = img_id
    if best_score == 0:
        return None
    return best_img


###############################################################################
# 5. 计算每一轮的指标
###############################################################################

def analyze_turn_row(
    question: str,
    answer: str,
    image_signatures: Dict[int, List[str]]
) -> dict:
    """
    返回这一轮的所有指标：
      - target_img
      - predicted_img
      - focus_correct
      - image_confusion
      - hallucination_level
      - truncation_hit
      - answer_len_chars / tokens
    """
    target_img = extract_target_image_id(question)
    predicted_img = infer_image_from_answer(answer, image_signatures)

    # focus_correct: 模型讲的图 == 题目问的图？
    # image_confusion: 模型讲的图 != 题目问的图？
    if target_img is not None:
        if predicted_img is None:
            focus_correct = 0
            image_confusion = 0  # 没猜到别的图，也不能算混淆
        else:
            focus_correct = 1 if (predicted_img == target_img) else 0
            image_confusion = 1 if (predicted_img != target_img) else 0
    else:
        # 题目没点名具体哪张图 => 我们不给它扣混淆
        focus_correct = None
        image_confusion = 0

    # 幻觉级别
    hallu_level = get_hallucination_level(answer)

    # 是否截断
    truncation_hit = 1 if "[TRUNCATED?]" in answer else 0

    # 长度
    ans_len_chars = len(answer.strip())
    ans_len_tokens = len(answer.strip().split())

    out = {
        "target_img": target_img,
        "predicted_img": predicted_img,
        "focus_correct": focus_correct,
        "image_confusion": image_confusion,
        "hallucination_level": hallu_level,
        "truncation_hit": truncation_hit,
        "answer_len_chars": ans_len_chars,
        "answer_len_tokens": ans_len_tokens,
    }
    return out


###############################################################################
# 6. 误差传播分析
#    - first_error_turn: 第一次出现严重错误的位置
#    - propagation_len: 之后连续多少轮保持“严重错误状态”
#
#    严重错误定义 = image_confusion==1 或 hallucination_level==2
###############################################################################

def compute_error_propagation(df: pd.DataFrame) -> Tuple[Optional[int], int]:
    serious = (df["image_confusion"] == 1) | (df["hallucination_level"] == 2)

    first_error_turn = None
    for idx, val in enumerate(serious.tolist(), start=1):
        if val:
            first_error_turn = idx
            break

    if first_error_turn is None:
        return None, 0

    # 从 first_error_turn 开始往后，连续多少轮还是 True
    prop_len = 0
    for k in range(first_error_turn - 1, len(df)):
        if serious.iloc[k]:
            prop_len += 1
        else:
            break

    return first_error_turn, prop_len


###############################################################################
# 7. 主函数
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="原始对话CSV (单模型×单样本×多turn)"
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="输出分析后的CSV；默认在原文件名后加 _analysis.csv"
    )
    parser.add_argument(
        "--sample-signature",
        type=str,
        default="sample8",
        help="用哪个签名词典（目前只有sample8，可以后扩展）"
    )
    args = parser.parse_args()

    # 选择签名词典（以后你可以按不同样本加分支）
    if args.sample_signature.lower() == "sample8":
        image_signatures = IMAGE_SIGNATURES_SAMPLE8
    else:
        # 如果你有别的样本，把别的 dict 塞在这里
        image_signatures = IMAGE_SIGNATURES_SAMPLE8
        print("[warn] using sample8 signatures as fallback")

    # 读CSV
    df_raw = pd.read_csv(args.csv)

    # 按 turn 排一下防乱序
    df_raw = df_raw.sort_values(by=["turn"]).reset_index(drop=True)

    # 周期性解析
    records = []
    for _, row in df_raw.iterrows():
        prompt = str(row["prompt"])
        answer = str(row["answer"])
        turn = int(row["turn"])

        question_text = extract_question_from_prompt(prompt)
        metrics = analyze_turn_row(
            question=question_text,
            answer=answer,
            image_signatures=image_signatures,
        )
        rec = {
            "turn": turn,
            "question": question_text,
            "answer": answer,
        }
        rec.update(metrics)
        records.append(rec)

    df_metrics = pd.DataFrame(records)

    # 误差传播
    first_error_turn, propagation_len = compute_error_propagation(df_metrics)

    print("===== Error Propagation Summary =====")
    print(f"first_error_turn       : {first_error_turn}")
    print(f"propagation_len (turns): {propagation_len}")
    print("=====================================")

    # 保存
    out_csv = args.out_csv
    if out_csv is None:
        if args.csv.endswith(".csv"):
            out_csv = args.csv[:-4] + "_analysis.csv"
        else:
            out_csv = args.csv + "_analysis.csv"

    df_metrics.to_csv(out_csv, index=False)
    print(f"[saved] per-turn analysis -> {out_csv}")


if __name__ == "__main__":
    main()
