# pilot_blip2_hpc_v3.py
import os
import io
import csv
import math
import argparse
from typing import List, Tuple, Optional
import re

from datasets import load_dataset
from huggingface_hub import hf_hub_download
import torch
from PIL import Image as PILImage
from PIL.Image import Image as PILImageType
from PIL import ImageDraw, ImageFont

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import BitsAndBytesConfig



# ===== Stopping: stop after 1 complete sentence, but only if long enough =====
class StopAfterOneSentence(StoppingCriteria):
    """
    每步查看最后一段生成文本；当长度>=min_chars 且以 .!?。！？ 结束时，提前停止。
    避免模型刚输出一个“.”就被误判停止。
    """
    def __init__(self, tokenizer, min_chars=28):
        super().__init__()
        self.tok = tokenizer
        self.min_chars = min_chars

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        tail = input_ids[0].tolist()[-96:]
        text = self.tok.decode(tail, skip_special_tokens=True)
        # 去除看不见的空白
        t = text.strip()
        if len(t) < self.min_chars:
            return False
        # 至少包含一个字母/数字（避免仅标点）
        if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", t):
            return False
        return bool(re.search(r"[.!?。！？]\s*$", t))


# --------------------------- HF dataset config ---------------------------
REPO_ID = "laolao77/MMDU"
ZIP_NAME = "mmdu_pics.zip"
EXTRACT_DIR = "./_mmdu_pics"

# --------------------------- utils: data ---------------------------
def ensure_mmdu_pics_local() -> str:
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    zip_path = hf_hub_download(repo_id=REPO_ID, filename=ZIP_NAME, repo_type="dataset")
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        need = True
        try_sample = next((n for n in zf.namelist() if n.lower().endswith((".jpg", ".png"))), None)
        if try_sample:
            testp = os.path.join(EXTRACT_DIR, try_sample)
            if os.path.exists(testp):
                need = False
        if need:
            zf.extractall(EXTRACT_DIR)
    root = os.path.join(EXTRACT_DIR, "mmdu_pics")
    return os.path.abspath(root if os.path.isdir(root) else EXTRACT_DIR)

def map_to_local_paths(paths: List[str], local_root: str) -> List[str]:
    out = []
    for p in paths:
        sub = p.lstrip("/")
        rel = os.path.relpath(sub, "mmdu_pics")
        out.append(os.path.join(local_root, rel))
    return out

def safe_open_image(x) -> PILImageType:
    if isinstance(x, PILImageType):
        return x.convert("RGB")
    elif isinstance(x, str):
        return PILImage.open(x).convert("RGB")
    elif isinstance(x, dict) and "path" in x:
        return PILImage.open(x["path"]).convert("RGB")
    else:
        raise ValueError(f"Unexpected image entry: type={type(x)}, value={x}")

# --------------------------- utils: panel ---------------------------
def load_font(size: int = 18):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()

def draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font, fill=(255, 255, 255)):
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 4
    draw.rectangle([bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad], fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=fill)

def make_image_panel(
    imgs: List[PILImageType],
    labels: List[str],
    tile: int = 384,
    cols: int = 3,
) -> PILImageType:
    assert len(imgs) == len(labels) and len(imgs) > 0
    font = load_font(18)

    resized = []
    for im in imgs:
        w, h = im.size
        if w >= h:
            nh = tile
            nw = int(w * nh / h)
        else:
            nw = tile
            nh = int(h * nw / w)
        im_r = im.resize((nw, nh), PILImage.BICUBIC)
        left = max(0, (nw - tile) // 2)
        top = max(0, (nh - tile) // 2)
        im_r = im_r.crop((left, top, left + tile, top + tile))
        resized.append(im_r)

    rows = math.ceil(len(resized) / cols)
    panel = PILImage.new("RGB", (cols * tile, rows * tile), (32, 32, 32))
    draw = ImageDraw.Draw(panel)

    for idx, (im, lab) in enumerate(zip(resized, labels)):
        r = idx // cols
        c = idx % cols
        x0 = c * tile
        y0 = r * tile
        panel.paste(im, (x0, y0))
        draw_label(draw, (x0 + 6, y0 + 6), lab, font)

    return panel

# --------------------------- utils: prompt ---------------------------
def shorten_name(path: str, keep: int = 24) -> str:
    bn = os.path.basename(path)
    return bn if len(bn) <= keep else (bn[:keep - 3] + "...")

def extract_focus_index(text: str, max_n: int) -> Optional[int]:
    ks = [int(m.group(1)) for m in re.finditer(r"image\s*(\d+)", text, flags=re.I)]
    ks = [k for k in ks if 1 <= k <= max_n]
    return (ks[-1] - 1) if ks else None

def build_prompt_multimg(short_names, recent_ctx, question, keep_ctx=2) -> str:
    guide = (
        f"You are given {len(short_names)} images as one panel. "
        "When I say ImageK, I mean the K-th tile in row-major order (left→right, top→bottom). "
        "Answer in ONE short, self-contained sentence. "
        "Do NOT list images, do NOT write labels like 'ImageK:' or 'Image 3:', and do NOT restate file names."
        "If multiple ImageK appear, answer about the **last mentioned** K only (do not compare images)."
        "If the question mentions a specific ImageK, answer about that image only; "
        "if multiple ImageK are mentioned, answer about the **last mentioned** one only; "
        "if none is mentioned, pick the single most relevant image and answer only about it."
    )
    parts = [guide]
    if recent_ctx:
        for i, s in enumerate(recent_ctx[-keep_ctx:], 1):
            parts.append(f"[Context {i}] {s}")
    parts.append(f"Q: {question}")
    parts.append("A:")
    return "\n".join(parts)

def make_bad_words_ids(tokenizer) -> list[list[int]]:
    bad_strs = [
        "Image list", "Image List", "Images:", "Image:", "Context:", "Question:", "Answer:",
        "Good example", "Bad example", "DO NOT", "Do NOT", "Example:",
        "<ImageHere>", "Image Here",
        "Image1:", "Image2:", "Image3:", "Image4:", "Image5:",
        "Image 1:", "Image 2:", "Image 3:", "Image 4:", "Image 5:",
        "Q:", "A:"
    ]
    out = []
    for s in bad_strs:
        ids = tokenizer(s, add_special_tokens=False).input_ids
        if ids:
            out.append(ids)
    return out

def strip_echo(prompt: str, text: str) -> str:
    s = text.strip()
    if prompt and s.startswith(prompt[:120]):
        s = s[len(prompt):].lstrip()
    for tok in ["Question:", "Context:", "Images:", "Image list:", "Answer:", "Short answer:", "Long answer:"]:
        idx = s.find(tok)
        if 0 <= idx < 12:
            s = s[idx + len(tok):].lstrip()
    for stop in ["\nQuestion:", "\nContext:", "\nImages:", "\nImage list:", "\n[", "\n- Image"]:
        cut = s.find(stop)
        if cut > 10:
            s = s[:cut].rstrip()
    return s

def make_turn_bad_words_ids(tokenizer, n_imgs: int) -> list[list[int]]:
    ban_strs = ["Q:", "Q :", "\nQ:", "\nQ :", "Question:", "Context:", "Image list:", "Image List:"]
    for k in range(n_imgs + 1, n_imgs + 16):
        ban_strs.extend([f"Image{k}", f"Image {k}", f"Image{k}:", f"Image {k}:"])
    out = []
    for s in ban_strs:
        ids = tokenizer(s, add_special_tokens=False).input_ids
        if ids:
            out.append(ids)
    return out

def clean_answer(text: str, prompt: str) -> str:
    s = (text or "").strip()

    if s.startswith("A:"):
        s = s[2:].lstrip()

    if prompt:
        s = s.replace(prompt[:200], "").strip()

    # 占位符、文件名
    s = re.sub(r"<[^>]*>", " ", s)
    s = re.sub(r"\b[\w\-]+\.(jpg|jpeg|png|gif|webp)\b", " ", s, flags=re.I)

    # 清理模板/标签
    kill = [
        r"\b(image\s*list|images?|context|question|answer)\s*:\s*",
        r"\bgood example\b", r"\bbad example\b", r"\bexample\s*:\s*",
        r"\bdo\s*not\b", r"\bimage\s*\d+\s*:\s*",
        r"\bq\s*:\b", r"\ba\s*:\b",
    ]
    for pat in kill:
        s = re.sub(pat, " ", s, flags=re.I)

    # 长数字/带下划线编号噪声
    s = re.sub(r"\b(?=\w*\d\w*\d\w*\d)\w+\b", " ", s)
    s = re.sub(r"\b\w*[_\-]\w*\d\w*\b", " ", s)

    # 删除纯编号行
    lines = [ln.strip() for ln in s.splitlines()]
    lines = [ln for ln in lines if not re.fullmatch(r"[-•*]?\s*\d+[.)]?", ln)]
    s = " ".join(lines)

    # 折叠空白
    s = re.sub(r"\s+", " ", s).strip()

    # 若仍出现续问，截断到前半段
    s = re.split(r"\b(?:q\s*:|question\s*:)\b", s, flags=re.I)[0].strip()

    # 只保留一句
    parts = re.split(r"(?<=[.!?。！？])\s+", s)
    s = (parts[0] if parts and parts[0] else s).strip()

    # 仅标点视为无效答案
    if not s or re.fullmatch(r"[.\u3002!！?？…]+", s):
        return "(no answer)"

    return s


# --------------------------- main ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Salesforce/blip2-opt-6.7b")
    ap.add_argument("--results", type=str, default="pilot_results_v3.csv")
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--hist-max", type=int, default=2)

    ap.add_argument("--max-images-per-turn", type=int, default=5)
    ap.add_argument("--panel-tile-size", type=int, default=384)
    ap.add_argument("--panel-cols", type=int, default=3)

    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--beams", type=int, default=3)

    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device-map", type=str, default="auto")
    ap.add_argument("--use-history", dest="use_history", action="store_true",
                help="Use short turn-level history summaries in the prompt.")

    args = ap.parse_args()

    local_root = ensure_mmdu_pics_local()
    print("Local images root:", local_root)

    ds = load_dataset(REPO_ID, split="train").select(range(args.num_samples))
    ds = ds.map(lambda ex: {"image_local": map_to_local_paths(ex["image"], local_root)})

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    print(f"Using device_map={args.device_map}, dtype={torch_dtype}")

    processor = Blip2Processor.from_pretrained(args.model)
    bnb_cfg = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,  # 允许把部分模块以 FP32 放到 CPU
    )

    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model,
        device_map=args.device_map,             # 建议 "auto"
        quantization_config=bnb_cfg,            # ← 新增：用 quantization_config 而不是 load_in_8bit
        low_cpu_mem_usage=True,                 # 边读边释放，降低峰值内存
        offload_folder="./offload",             # 磁盘换页目录（会自动创建）
        torch_dtype=None,                       # 量化时 dtype 不用再指定（避免冲突）
    )


    tok = processor.tokenizer
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id
    bad_words_ids = make_bad_words_ids(tok)

    try:
        max_pos = getattr(model.language_model.config, "max_position_embeddings", 2048)
    except Exception:
        max_pos = 2048
    margin = 8
    ctx_budget = max(max_pos - args.max_new_tokens - margin, 256)
    print(f"[ctx] max_pos={max_pos}  max_new={args.max_new_tokens}  budget={ctx_budget}")

    # —— 解码参数（统一在每轮覆盖 bad_words_ids & stopper）——
    gen_base = dict(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=4,                     # 放宽到 4，避免被迫续太长
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        use_cache=True,
        num_beams=max(1, args.beams),
        length_penalty=0.05,
        early_stopping=True,
    )
    if args.do_sample:
        gen_base.update(dict(
            do_sample=True, temperature=args.temperature, top_p=args.top_p,
            no_repeat_ngram_size=4, repetition_penalty=1.1,
        ))
    else:
        gen_base.update(dict(
            do_sample=False, no_repeat_ngram_size=6, repetition_penalty=1.25,
        ))

    # CSV 输出
    with open(args.results, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "turn", "img_idxes", "image_files", "prompt", "answer"])

        for i in range(len(ds)):
            ex = ds[i]
            conv = ex["conversations"]
            paths: List[str] = ex["image_local"]

            print("\n" + "=" * 60)
            print(f"Sample {i} | #images={len(paths)}  #turns={len(conv)}")

            sel_paths = paths[: args.max_images_per_turn]
            sel_imgs = [safe_open_image(p) for p in sel_paths]
            short_names = [shorten_name(p) for p in sel_paths]
            idxes_str = ";".join(str(k) for k in range(len(sel_paths)))
            files_str = ";".join(sel_paths)

            panel = make_image_panel(
                imgs=sel_imgs,
                labels=[f"Image{k+1}" for k in range(len(sel_imgs))],
                tile=args.panel_tile_size,
                cols=args.panel_cols,
            )

            user_utts = [t["value"] for t in conv if t.get("from") == "user"]
            max_turns = min(args.max_turns, len(user_utts))
            history_summaries: List[str] = []

            for t in range(max_turns):
                q = user_utts[t]

                prompt = build_prompt_multimg(
                    short_names=short_names,
                    recent_ctx=(history_summaries if args.use_history else []),
                    question=q,
                    keep_ctx=args.hist_max,
                )

                n_tok = len(tok(prompt, add_special_tokens=False).input_ids)
                print(f"\n[Turn {t+1}] | images={len(sel_imgs)} -> panel")
                print("Prompt (head):", prompt[:120].replace("\n", " / "))
                print(f"[tokens] prompt={n_tok}  budget={ctx_budget}")

                focus_idx = extract_focus_index(q, max_n=len(sel_imgs))
                infer_img = sel_imgs[focus_idx] if focus_idx is not None else panel

                bad_words_ids_turn = make_turn_bad_words_ids(tok, n_imgs=len(sel_imgs))

                inputs = processor(
                    images=infer_img,
                    text=prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=ctx_budget,
                )
                for k in list(inputs.keys()):
                    if isinstance(inputs[k], torch.Tensor):
                        inputs[k] = inputs[k].to(model.device)

                # 合成本轮的生成参数
                gen_kwargs = dict(gen_base)
                gen_kwargs["bad_words_ids"] = bad_words_ids_turn  # 动态封禁
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [StopAfterOneSentence(tok, min_chars=28)]
                )

                output_ids = model.generate(**inputs, **gen_kwargs)

                # —— 精准判断是否打满 —— 
                inp_len = inputs.get("input_ids", None)
                if inp_len is not None and isinstance(inp_len, torch.Tensor):
                    new_len = output_ids.shape[-1] - inp_len.shape[-1]
                else:
                    new_len = output_ids.shape[-1]

                ended_with_eos = (tok.eos_token_id is not None) and (output_ids[0, -1].item() == tok.eos_token_id)
                hit_limit = new_len >= (args.max_new_tokens - 1)

                raw_text = tok.decode(output_ids[0], skip_special_tokens=True)
                raw_text = strip_echo(prompt, raw_text)
                answer = clean_answer(raw_text, prompt)

                if hit_limit and not ended_with_eos:
                    answer = answer.rstrip() + " [TRUNCATED?]"

                print("Answer (head):", (answer[:140] + ("..." if len(answer) > 140 else "")))
                writer.writerow([i, t+1, idxes_str, files_str, prompt, answer])

    print(f"\n✅ Results saved to {args.results}")
    print("Tip: If you still hit context budget, reduce --max-images-per-turn or --panel-tile-size or --hist-max.")

if __name__ == "__main__":
    main()
