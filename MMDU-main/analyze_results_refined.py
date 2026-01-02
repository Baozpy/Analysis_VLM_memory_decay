import re
import csv
import argparse
from typing import List, Optional, Tuple


IMG_REF_RE = re.compile(r"Image\s*([1-9]\d*)", re.IGNORECASE)

def extract_all_images(text: str) -> List[int]:
    """
    从一段文本里抽取所有形式类似 'Image3', 'Image 2' 的引用，转成 int。
    只要匹配到数字就收集。大小写不敏感。
    """
    if text is None:
        return []
    return [int(m.group(1)) for m in IMG_REF_RE.finditer(text)]


def get_target_image(question: str) -> Tuple[Optional[int], List[int]]:
    """
    从 question 里找本轮应该关注的图。
    规则（和我们之前分析一致）：
    - 找到所有 "ImageK"
    - 如果有多张，就用最后提到的那一张作为本轮主关注对象
      （和你之前“last mentioned wins”的标注一致）
    - 返回 (target_img, unique_imgs_in_question)

    如果问题里完全没出现 ImageK，返回 (None, []).
    """
    imgs = extract_all_images(question)
    if not imgs:
        return None, []
    return imgs[-1], list(dict.fromkeys(imgs))  # 去重保持顺序


def get_predicted_image(answer: str) -> Optional[int]:
    """
    尝试直接从 answer 里面解析它在说哪张图：
    - 如果它明确说了 "Image2 ... Image3 ..." 我们认为最后提到的是主语
    - 如果完全没提，就先返回 None（后面还有 fallback 逻辑）
    """
    imgs = extract_all_images(answer)
    if not imgs:
        return None
    return imgs[-1]


def fallback_predicted_if_single_target(
    predicted_img: Optional[int],
    target_img: Optional[int],
    imgs_in_q: List[int],
    answer: str,
) -> Optional[int]:
    """
    这是新的关键补丁。

    目标：避免把“模型明明在讲正确的图，但没重复说‘Image1’”判成忘记。

    逻辑：
    1. 如果我们已经从回答里直接识别到了 predicted_img（也就是回答自己说了 ImageK），
       那就尊重这个识别，不改。
    2. 否则（predicted_img is None）：
       - 如果问题里只提到了一张图 (len(imgs_in_q) == 1)，
       - 而且我们本轮有明确的 target_img，
       - 而且回答里也没有提到别的 image（这里其实是 implied by predicted_img is None），
         那我们就假定它还在讲那张 target_img。

    这样可以修正 Qwen/LLaVA 在 turn1 那种“自然口语式回答”，
    但不会掩盖真正的cross-image串台。
    """
    if predicted_img is not None:
        return predicted_img

    if target_img is None:
        return None

    # 问题里是否只点名了一张图？
    if len(imgs_in_q) == 1:
        # 回答里没有出现其它 ImageK（因为 predicted_img is None）
        # -> 我们合理地认为它还在专注同一张图
        return target_img

    # 多图对比类问题("Compare Image1 and Image4...")，我们不自动帮它选
    return None


def compute_focus_and_confusion(target_img: Optional[int], predicted_img: Optional[int]) -> Tuple[int, int]:
    """
    返回 (focus_correct, image_confusion)

    focus_correct = 1 当且仅当 predicted_img == target_img 且二者都存在
    image_confusion = 1 当 predicted_img 存在、target_img 存在、且不相等
                     否则 0

    注意：如果 predicted_img 为空(None)，focus_correct=0, image_confusion=0
    （因为我们不知道它跑哪了；真混淆是在它明确提了别的图时才标1）
    """
    if target_img is not None and predicted_img is not None:
        if predicted_img == target_img:
            return 1, 0
        else:
            return 0, 1
    # 没法判断是否混淆，只能说“不确认对焦成功”
    return 0, 0


def compute_truncation_hit(answer: str) -> int:
    """
    很简单的启发式：如果答案最后带了 [TRUNCATED?]，我们记1。
    """
    if answer is None:
        return 0
    return 1 if "[TRUNCATED" in answer else 0


def compute_hallucination_level(answer: str) -> int:
    """
    这里我先保留一个轻量占位逻辑，默认返回0。
    你之后如果想把 BLIP2 的“华盛顿纪念堂式幻觉”单独打成2，也可以在这里做regex扩展。

    举例扩展思路（可选）：
      if it starts inventing 'the first president of the united states' in a tomb
      => return 2
    """
    if answer is None:
        return 0
    halluc_keywords_lv2 = [
        "first president of the united states",
        "washington",
        "eagle",
        "gold dome",
    ]
    for kw in halluc_keywords_lv2:
        if kw.lower() in answer.lower():
            return 2
    # 轻度幻觉可以是大段故事化幻想，这里不细分，统一0
    return 0


def tokenize_len(answer: str) -> int:
    """
    非严格 token 计数，简单用空格 split 模拟。
    这样做的目的是能看相对长度趋势，而不是绝对精确到 tokenizer。
    """
    if answer is None:
        return 0
    return len(answer.strip().split())


def analyze_dialog(csv_in: str, csv_out: str):
    rows_in = []
    with open(csv_in, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows_in.append(r)

    out_rows = []
    for r in rows_in:
        turn = r.get("turn")
        question = r.get("question", "") or r.get("prompt", "")
        answer = r.get("answer", "")

        # 1. 解析本轮应该关注哪张图（target_img）
        target_img, imgs_in_q = get_target_image(question)

        # 2. 直接从回答里看它说它在看哪张图
        predicted_direct = get_predicted_image(answer)

        # 3. fallback：单图提问 + 回答没换图时，自动承认它还在讲同一张
        predicted_final = fallback_predicted_if_single_target(
            predicted_direct,
            target_img,
            imgs_in_q,
            answer,
        )

        # 4. 计算 focus_correct / image_confusion
        focus_correct, image_confusion = compute_focus_and_confusion(
            target_img, predicted_final
        )

        # 5. 其它指标
        truncation_hit = compute_truncation_hit(answer)
        halluc_level = compute_hallucination_level(answer)
        ans_len_chars = len(answer)
        ans_len_tokens = tokenize_len(answer)

        out_rows.append({
            "turn": turn,
            "question": question,
            "answer": answer,
            "target_img": target_img if target_img is not None else "",
            "predicted_img": predicted_final if predicted_final is not None else "",
            "focus_correct": focus_correct,
            "image_confusion": image_confusion,
            "hallucination_level": halluc_level,
            "truncation_hit": truncation_hit,
            "answer_len_chars": ans_len_chars,
            "answer_len_tokens": ans_len_tokens,
        })

    # 写出 refined 版，不覆盖你之前的 _analysis.csv
    fieldnames = [
        "turn",
        "question",
        "answer",
        "target_img",
        "predicted_img",
        "focus_correct",
        "image_confusion",
        "hallucination_level",
        "truncation_hit",
        "answer_len_chars",
        "answer_len_tokens",
    ]

    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="原始20轮结果CSV，至少包含 turn,question,answer 三列")
    ap.add_argument("--out", required=True, help="输出的新打分文件，比如 qwen_sample8_analysis_refined.csv")
    args = ap.parse_args()

    analyze_dialog(args.input, args.out)
