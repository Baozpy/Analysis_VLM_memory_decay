# pilot_multimodel_v1.py
import os
import csv
import math
import argparse
from typing import List, Tuple, Optional
import re
import inspect
import traceback

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image as PILImage
from PIL.Image import Image as PILImageType
from PIL import ImageDraw, ImageFont
import gc  

# 用专用类跑 LLaVA v1.6，避免误走 llava-next 的实现
from transformers import LlavaForConditionalGeneration, LlavaProcessor

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
        rel = os.path.relpath(sub, "mmdu_pics")  # /mmdu_pics/xxx -> xxx
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
    """把多张图做成网格面板，左上角标 'ImageK'。"""
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

# --------------------------- utils: prompt & clean ---------------------------
def shorten_name(path: str, keep: int = 24) -> str:
    bn = os.path.basename(path)
    return bn if len(bn) <= keep else (bn[:keep - 3] + "...")

def extract_focus_index(text: str, max_n: int) -> Optional[int]:
    """抓到所有 ImageK，取最后一个合法 K（1-based 转 0-based）。"""
    ks = [int(m.group(1)) for m in re.finditer(r"image\s*(\d+)", text, flags=re.I)]
    ks = [k for k in ks if 1 <= k <= max_n]
    return (ks[-1] - 1) if ks else None

def build_prompt_multimg(short_names, recent_ctx, question, keep_ctx=2) -> str:
    guide = (
        f"You are given {len(short_names)} images as one panel. "
        "When I say ImageK, I mean the K-th tile in row-major order (left→right, top→bottom). "
        "Answer in ONE short, self-contained sentence. "
        "Do NOT list images, do NOT write labels like 'ImageK:' or 'Image 3:', and do NOT restate file names."
        "If multiple ImageK appear, answer about the **last mentioned** K only (do not compare images). "
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


def strip_echo(prompt: str, text: str) -> str:
    s = (text or "").strip()

    # Qwen 模板标记清理
    s = re.sub(r"<\|im_start\|>\s*(system|user|assistant)\s*", "", s, flags=re.I)
    s = s.replace("<|im_end|>", " ").strip()

    # 如果包含角色模板，截到最后一个 "assistant" 之后
    for tag in ["ASSISTANT:", "Assistant:", "assistant:"]:
        idx = s.rfind(tag)
        if idx != -1:
            s = s[idx + len(tag):].lstrip()

    # 去掉行首角色头
    s = re.sub(r"^\s*(USER|User|user|SYSTEM|System|system|ASSISTANT|Assistant|assistant)\s*[:：]\s*", "", s)

    # 老模板回声裁剪
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

    s = re.sub(r"\s+", " ", s).strip()
    parts = re.split(r"(?<=[.!?。！？])\s+", s)
    s = (parts[0] if parts and parts[0] else s).strip()
    return s if s else "(no answer)"




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
    s = re.sub(r"<[^>]*>", " ", s)
    s = re.sub(r"\b[\w\-]+\.(jpg|jpeg|png|gif|webp)\b", " ", s, flags=re.I)
    kill_phrases = [
        r"\b(image\s*list|images?|context|question|answer)\s*:\s*",
        r"\bgood example\b", r"\bbad example\b", r"\bexample\s*:\s*",
        r"\bdo\s*not\b",
        r"\bimage\s*\d+\s*:\s*",
        r"\bq\s*:\b", r"\ba\s*:\b",
    ]
    for pat in kill_phrases:
        s = re.sub(pat, " ", s, flags=re.I)
    s = re.sub(r"\b(?=\w*\d\w*\d\w*\d)\w+\b", " ", s)   # long id-like
    s = re.sub(r"\b\w*[_\-]\w*\d\w*\b", " ", s)
    lines = [ln.strip() for ln in s.splitlines()]
    lines = [ln for ln in lines if not re.fullmatch(r"[-•*]?\s*\d+[.)]?", ln)]
    s = " ".join(lines)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.split(r"\b(?:q\s*:|question\s*:)\b", s, flags=re.I)[0].strip()
    parts = re.split(r"(?<=[.!?。！？])\s+", s)
    s = (parts[0] if parts and parts[0] else s).strip()
    return s if s else "(no answer)"

# --------------------------- VLM wrapper ---------------------------
from dataclasses import dataclass

def _is_llava_next(model) -> bool:
    mod = getattr(model.__class__, "__module__", "").lower()
    name = getattr(model.__class__, "__name__", "").lower()
    return ("llava_next" in mod) or ("llavanext" in name)

@dataclass
class SimpleVLM:
    kind: str
    model: object
    processor: object
    tokenizer: object

    def tokenize_len(self, text: str) -> int:
        tok = self.tokenizer or getattr(self.processor, "tokenizer", None)
        if tok is None:
            return len(text)
        return len(tok(text, add_special_tokens=False).input_ids)

    def _calc_image_num_patches(self, pixel_values: torch.Tensor) -> list[int] | None:
        try:
            if pixel_values.ndim == 5:
                if pixel_values.shape[1] == 1:
                    pixel_values = pixel_values[:, 0, ...]
                else:
                    pixel_values = pixel_values.mean(dim=1)
            if pixel_values.ndim != 4:
                return None
            B, _, H, W = pixel_values.shape
            patch = getattr(
                getattr(getattr(self.model, "vision_tower", None), "config", None),
                "patch_size",
                14,
            )
            per_img = (H // patch) * (W // patch)
            return [int(per_img)] * int(B)
        except Exception:
            return None

    def prepare_inputs(self, image, prompt: str, ctx_budget: int):
        kind = (self.kind or "").lower()

        # ---------- BLIP-2 ----------
        if "blip2" in kind:
            return self.processor(
                images=image, text=prompt,
                return_tensors="pt", padding=False, truncation=True, max_length=ctx_budget
            )

        # ---------- LLaVA v1.6：专用写法（避免乱码/错位） ----------
        if kind == "llava_v16":
            import re
            prompt = re.sub(r"<image.*?>", "", prompt, flags=re.IGNORECASE).strip()
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }]
            # 1) 先得到纯文本模板
            chat_text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,          # 重要：返回字符串模板
            )
            # 2) 交给 LlavaProcessor 一次性处理 text+image
            proc = self.processor(
                text=[chat_text],        # 注意：是列表
                images=[image],          # 注意：也是列表
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=ctx_budget,
            )
            # 此路径返回的 pixel_values 应为 [B, 3, H, W] 四维
            return proc




        # ---------- 其他 LLaVA（含 next） ----------
        if "llava" in kind:
            import re
            prompt = re.sub(r"<image.*?>", "", prompt, flags=re.IGNORECASE).strip()
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }]
            chat_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            proc = self.processor(
                text=chat_text,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=ctx_budget,
            )
            return proc

        # ---------- Qwen2-VL ----------
        # ---------- Qwen2-VL ----------
        if "qwen2-vl" in kind or "qwen/qwen2-vl" in kind:
            import re
            prompt = re.sub(r"<image.*?>", "", prompt, flags=re.IGNORECASE).strip()
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }]

            # 1) 用 chat_template 先拿纯文本模版（不要直接 tokenize）
            chat_text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            # # 2) 交给 AutoProcessor 一次性处理 text+image（会给出 4D pixel_values）
            # proc = self.processor(
            #     text=[chat_text],
            #     images=[image],
            #     return_tensors="pt",
            #     padding=True,
            #     truncation=True,
            #     max_length=ctx_budget,
            # )
            from PIL import Image as PILImage

            def _resize_longest_edge(im: PILImage.Image, longest: int = 512) -> PILImage.Image:
                w, h = im.size
                m = max(w, h)
                if m <= longest:
                    return im
                scale = float(longest) / float(m)
                nw, nh = int(w * scale), int(h * scale)
                return im.resize((nw, nh), PILImage.BICUBIC)

            # …… 在构造 messages / chat_text 之后：
            image_small = _resize_longest_edge(image, longest=512)  # 或 448/480，512 更安全

            proc = self.processor(
                text=[chat_text],
                images=[image_small],   # ← 用缩过的图
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=ctx_budget,
            )


            # 3) 极少数情况下会给 5D，保险压回 4D
            if "pixel_values" in proc and isinstance(proc["pixel_values"], torch.Tensor) and proc["pixel_values"].ndim == 5:
                pv = proc["pixel_values"]
                if pv.shape[2] == 3:  # [B,N,3,H,W]
                    proc["pixel_values"] = pv[:, 0, ...]
                elif pv.shape[1] == 3:  # [B,3,C,H,W]（罕见）
                    proc["pixel_values"] = pv[:, :, 0, ...]

            return proc


        

def load_multimodel(model_name: str, dtype, device_map="auto") -> SimpleVLM:
    name_lower = model_name.lower()

    # ---------- BLIP-2 ----------
    if "blip2" in name_lower:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device_map
        )
        tok = processor.tokenizer
        return SimpleVLM(kind="blip2", model=model, processor=processor, tokenizer=tok)

    # ---------- LLaVA v1.6: 强制用官方类 ----------
    if "llava-hf/llava-v1.6" in name_lower or "llava-v1.6" in name_lower:
        processor = LlavaProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        return SimpleVLM(kind="llava_v16", model=model, processor=processor,
                         tokenizer=getattr(processor, "tokenizer", None))

    # ---------- 其他 LLaVA（含 next） ----------
    if "llava" in name_lower:
        from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        kind = getattr(cfg, "model_type", "llava")
        if kind not in ("llava", "llava_next"):
            kind = "llava_next" if "llava_next" in type(model).__module__ else "llava"
        return SimpleVLM(kind=kind, model=model, processor=processor,
                         tokenizer=getattr(processor, "tokenizer", None))

    # ---------- Qwen2-VL ----------
    # ---------- Qwen2-VL ----------
    if "qwen2-vl" in name_lower or "qwen/qwen2-vl" in name_lower:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
        )
        tok = processor.tokenizer
        return SimpleVLM(kind="qwen2-vl", model=model, processor=processor, tokenizer=tok)


# --------------------------- safe generate ---------------------------
def _pack_basic_inputs(inputs: dict, model) -> dict:
    """
    保留核心键：对通用模型仅保留 input/attn/pixel；
    对 Qwen2-VL 还必须保留 image_grid_thw。
    同时把张量搬到模型设备并矫正 dtype（ID/Mask=long，像素=模型精度）。
    """
    keep = {}
    # 先拿到常见键
    for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
        if k in inputs:
            keep[k] = inputs[k]

    # 逐键处理 dtype/设备（区别对待）
    dev = model.device
    for k, v in list(keep.items()):
        if not isinstance(v, torch.Tensor):
            continue
        if k in ("input_ids", "attention_mask"):
            keep[k] = v.to(dev, dtype=torch.long)
        elif k == "pixel_values":
            # 对于 LLaVA：这是 [B,3,H,W]； 对于 Qwen2-VL：可能已是 [T,D]（保持原样）
            keep[k] = v.to(dev, dtype=getattr(model, "dtype", torch.bfloat16))
        elif k == "image_grid_thw":
            # Qwen2-VL 位置编码需要整数网格
            keep[k] = v.to(dev, dtype=torch.long)

    # 这些键容易在不同实现里触发不兼容，统一剥掉
    for bad in ("image_sizes", "image_num_patches", "num_image_tokens"):
        keep.pop(bad, None)

    return keep


def _compute_aux_from_pixel_values(model_inputs: dict, model):
    if "pixel_values" not in model_inputs:
        return None, None
    pv = model_inputs["pixel_values"]
    if not (isinstance(pv, torch.Tensor) and pv.ndim == 4):
        return None, None
    B, _, H, W = pv.shape
    patch_size = getattr(getattr(getattr(model, "vision_tower", None), "config", None), "patch_size", 14)
    n_p = (H // patch_size) * (W // patch_size)
    image_num_patches = [int(n_p)] * int(B)
    image_sizes = [(int(H), int(W))] * int(B)
    return image_num_patches, image_sizes

def safe_generate(model, base_inputs: dict, gen_kwargs: dict, debug: bool=False):
    def log(*a):
        if debug: print("[safe_generate]", *a)

    # ---------- v1.6 官方类：矫正 dtype + 干净直跑（最稳） ----------
    if model.__class__.__name__ == "LlavaForConditionalGeneration":
        mi = dict(base_inputs)
        dev = model.device
        if "input_ids" in mi and isinstance(mi["input_ids"], torch.Tensor):
            mi["input_ids"] = mi["input_ids"].to(dev, dtype=torch.long)
        if "attention_mask" in mi and isinstance(mi["attention_mask"], torch.Tensor):
            mi["attention_mask"] = mi["attention_mask"].to(dev, dtype=torch.long)
        if "pixel_values" in mi and isinstance(mi["pixel_values"], torch.Tensor):
            mi["pixel_values"] = mi["pixel_values"].to(dev, dtype=getattr(model, "dtype", torch.bfloat16))
            # 如果意外是 5D，压回 4D（保险）
            if mi["pixel_values"].ndim == 5:
                pv = mi["pixel_values"]
                if pv.shape[2] == 3:
                    mi["pixel_values"] = pv[:, 0, ...]
                elif pv.shape[1] == 3:
                    mi["pixel_values"] = pv[:, :, 0, ...]

        # 👉 关键：v1.6 不要把这些 kw 传下去
        mi.pop("image_sizes", None)
        mi.pop("image_num_patches", None)
        mi.pop("num_image_tokens", None)

        
        gk = dict(gen_kwargs)
        # 收敛为低显存、确定性
        gk["num_beams"] = 1
        gk["do_sample"] = False
        gk["use_cache"] = False
        gk.pop("early_stopping", None)
        gk.pop("length_penalty", None)
        # 禁词在 v1.6 上有时会误杀常见 token，先移除
        gk.pop("bad_words_ids", None)

        if debug:
            ks = ", ".join(sorted(mi.keys()))
            log(f"[v1.6] keys= {ks}")
            for k, v in mi.items():
                if isinstance(v, torch.Tensor):
                    log(f"  {k}: shape={tuple(v.shape)} device={v.device} dtype={v.dtype}")
            log(f"[v1.6] gen_kwargs: " +
                ", ".join(f"{k}={v}" for k, v in gk.items()
                          if k in ("num_beams","do_sample","use_cache","max_new_tokens","eos_token_id","pad_token_id")))

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            return model.generate(**mi, **gk)

    # ---------- 其他模型：三段式兜底 ----------
    mi = _pack_basic_inputs(base_inputs, model)
    if debug:
        ks = ", ".join(sorted(mi.keys()))
        log(f"keys={ks}")
        for k, v in mi.items():
            if isinstance(v, torch.Tensor):
                log(f"  {k}: shape={tuple(v.shape)} device={v.device} dtype={v.dtype}")

    # 读取 forward 签名，看是否能显式接收 image_num_patches
    try:
        fwd_sig = inspect.signature(model.forward)
        fwd_params = set(fwd_sig.parameters.keys())
    except Exception:
        fwd_sig, fwd_params = None, set()
    accepts_patches = ("image_num_patches" in fwd_params) or (fwd_sig is None)
    log(f"forward has image_num_patches? {accepts_patches}")

    def compute_aux_from_pv(mi_local):
        img_patches, img_sizes = _compute_aux_from_pixel_values(mi_local, model)
        log(f"computed patches={img_patches[:2] if img_patches else None} sizes_sample={img_sizes[0] if img_sizes else None}")
        return img_patches, img_sizes

    # try #1: clean
    try:
        with torch.no_grad():
            log("try #1: clean generate()")
            return model.generate(**mi, **gen_kwargs)
    except Exception:
        tb = "".join(traceback.format_exc())
        tbl = tb.lower()
        log("try #1 failed.")
        log("traceback head:", tb.splitlines()[-10:])

        patterns = ["image_num_patches", "split_with_sizes", "nonetype", "object is not iterable"]
        if not any(p in tbl for p in patterns):
            raise

        # try #2: add extras
        mi2 = dict(mi)
        img_patches, img_sizes = compute_aux_from_pv(mi2)
        extra = {}
        if accepts_patches and img_patches is not None:
            extra["image_num_patches"] = img_patches
        is_next = _is_llava_next(model)
        if is_next and (img_sizes is not None):
            extra["image_sizes"] = img_sizes

        log(f"try #2: retry with extras keys={list(extra.keys())}")
        try:
            with torch.no_grad():
                return model.generate(**mi2, **gen_kwargs, **extra)
        except Exception:
            tb2 = "".join(traceback.format_exc())
            tbl2 = tb2.lower()
            log("try #2 failed.")
            log("traceback head:", tb2.splitlines()[-10:])

            if ("not used by the model" in tbl2) or ("unexpected keyword argument" in tbl2):
                log("try #3: strip extras and retry clean again")
                with torch.no_grad():
                    return model.generate(**mi, **gen_kwargs)
            raise
    
    # ---------- Qwen2-VL：矫正 dtype + 干净直跑 ----------
    if model.__class__.__name__ == "Qwen2VLForConditionalGeneration":
        mi = dict(base_inputs)
        dev = model.device
        # ids/mask 必须是 long；像素用模型精度（bf16）
        if "input_ids" in mi and isinstance(mi["input_ids"], torch.Tensor):
            mi["input_ids"] = mi["input_ids"].to(dev, dtype=torch.long)
        if "attention_mask" in mi and isinstance(mi["attention_mask"], torch.Tensor):
            mi["attention_mask"] = mi["attention_mask"].to(dev, dtype=torch.long)
        if "pixel_values" in mi and isinstance(mi["pixel_values"], torch.Tensor):
            mi["pixel_values"] = mi["pixel_values"].to(dev, dtype=getattr(model, "dtype", torch.bfloat16))
            if mi["pixel_values"].ndim == 5:
                pv = mi["pixel_values"]
                if pv.shape[2] == 3:
                    mi["pixel_values"] = pv[:, 0, ...]
                elif pv.shape[1] == 3:
                    mi["pixel_values"] = pv[:, :, 0, ...]

        # 有些 Qwen 会把 image_sizes 透传到 Vision 塔，不吃；直接剥掉
        mi.pop("image_sizes", None)
        mi.pop("image_num_patches", None)
        mi.pop("num_image_tokens", None)

        gk = dict(gen_kwargs)
        # 统一低显存、确定性；Qwen 不稳定时也别传 bad_words_ids
        gk["num_beams"] = 1
        gk["do_sample"] = False
        gk["use_cache"] = False
        gk.pop("early_stopping", None)
        gk.pop("length_penalty", None)
        gk.pop("bad_words_ids", None)

        with torch.no_grad():
            return model.generate(**mi, **gk)


# --------------------------- main ---------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model", type=str, required=True,
#                     help="HF model id, e.g., Salesforce/blip2-opt-6.7b | llava-hf/llava-v1.6-vicuna-7b-hf | Qwen/Qwen2-VL-7B-Instruct")
#     ap.add_argument("--results", type=str, default="pilot_multimodel_results.csv")
#     ap.add_argument("--num-samples", type=int, default=16)
#     ap.add_argument("--max-turns", type=int, default=4)
#     ap.add_argument("--use-history", dest="use_history", action="store_true")
#     ap.add_argument("--no-use-history", dest="use_history", action="store_false")
#     ap.set_defaults(use_history=True)
#     ap.add_argument("--hist-max", type=int, default=2)

#     ap.add_argument("--max-images-per-turn", type=int, default=5)
#     ap.add_argument("--panel-tile-size", type=int, default=384)
#     ap.add_argument("--panel-cols", type=int, default=3)

#     ap.add_argument("--max-new-tokens", type=int, default=128)
#     ap.add_argument("--do-sample", action="store_true")
#     ap.add_argument("--temperature", type=float, default=0.7)
#     ap.add_argument("--top_p", type=float, default=0.9)
#     ap.add_argument("--beams", type=int, default=1)

#     ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
#     ap.add_argument("--device-map", type=str, default="auto")
#     ap.add_argument("--debug", action="store_true")
#     ap.add_argument("--start-sample", type=int, default=0,
#                     help="从第几个样本开始（0-based）")

#     args = ap.parse_args()

#     # ---------- Dataset ----------
#     local_root = ensure_mmdu_pics_local()
#     print("Local images root:", local_root)

#     ds_full = load_dataset(REPO_ID, split="train")

#     start = max(0, args.start_sample)
#     end = start + args.num_samples if args.num_samples is not None else len(ds_full)
#     end = min(end, len(ds_full))

#     ds = ds_full.select(range(start, end))
#     ds = ds.map(lambda ex: {"image_local": map_to_local_paths(ex["image"], local_root)})

#     print(f"[dataset] using samples [{start}, {end}) -> total {len(ds)}")

#     # ---------- dtype ----------
#     dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
#     torch_dtype = dtype_map[args.dtype]
#     print(f"Using device_map={args.device_map}, dtype={torch_dtype}")

#     # ---------- Load multimodal model ----------
#     vlm = load_multimodel(args.model, dtype=torch_dtype, device_map=args.device_map)
#     tok = vlm.tokenizer
#     eos_id = getattr(tok, "eos_token_id", None)
#     pad_id = getattr(tok, "pad_token_id", None)

#     # ---------- 显存稳态 ----------
#     torch.set_grad_enabled(False)
#     try:
#         torch.backends.cuda.matmul.allow_tf32 = True
#     except Exception:
#         pass

#     # ---------- 生成参数（单处定义） ----------
#     gen_kwargs = dict(
#         max_new_tokens=args.max_new_tokens,
#         use_cache=True,
#     )
#     if eos_id is not None:
#         gen_kwargs["eos_token_id"] = eos_id
#     if pad_id is not None:
#         gen_kwargs["pad_token_id"] = pad_id

#     if args.do_sample:
#         gen_kwargs.update(dict(
#             do_sample=True, temperature=args.temperature, top_p=args.top_p,
#             num_beams=max(1, args.beams),
#         ))
#     else:
#         gen_kwargs.update(dict(
#             do_sample=False,
#             num_beams=1,  # 避免 OOM
#         ))

#     # ---------- context budget ----------
#     try:
#         cfg = getattr(getattr(vlm.model, "language_model", vlm.model), "config", None)
#         max_pos = getattr(cfg, "max_position_embeddings", 2048) if cfg else 2048
#     except Exception:
#         max_pos = 2048
#     margin = 8
#     ctx_budget = max(max_pos - args.max_new_tokens - margin, 256)
#     print(f"[ctx] max_pos={max_pos}  max_new={args.max_new_tokens}  budget={ctx_budget}")

#     # ---------- CSV ----------
#     with open(args.results, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["sample_id", "turn", "img_idxes", "image_files", "prompt", "answer"])

#         for i in range(len(ds)):
#             ex = ds[i]
#             conv = ex["conversations"]
#             paths: List[str] = ex["image_local"]

#             print("\n" + "=" * 60)
#             print(f"Sample {i} | #images={len(paths)}  #turns={len(conv)}")

#             sel_paths = paths[: args.max_images_per_turn]
#             sel_imgs = [safe_open_image(p) for p in sel_paths]
#             short_names = [shorten_name(p) for p in sel_paths]
#             idxes_str = ";".join(str(k) for k in range(len(sel_paths)))
#             files_str = ";".join(sel_paths)

#             panel = make_image_panel(
#                 imgs=sel_imgs,
#                 labels=[f"Image{k+1}" for k in range(len(sel_imgs))],
#                 tile=args.panel_tile_size,
#                 cols=args.panel_cols,
#             )

#             user_utts = [t["value"] for t in conv if t.get("from") == "user"]
#             max_turns = min(args.max_turns, len(user_utts))
#             history_summaries: List[str] = []

#             for t in range(max_turns):
#                 q = user_utts[t]
#                 prompt = build_prompt_multimg(
#                     short_names=short_names,
#                     recent_ctx=(history_summaries if args.use_history else []),
#                     question=q,
#                     keep_ctx=args.hist_max,
#                 )

#                 # --- BLIP-2 专用：提示词更直接，减少过度约束 ---
#                 if getattr(vlm, "kind", "") and "blip2" in vlm.kind.lower():
#                     prompt = (
#                         f"You are given {len(short_names)} images as one panel. "
#                         "When I say ImageK, I mean the K-th tile in row-major order (left→right, top→bottom). "
#                         "Answer in one concise sentence.\n"
#                         f"Q: {q}\nA:"
#                     )

#                 n_tok = vlm.tokenize_len(prompt)
#                 print(f"\n[Turn {t+1}] | images={len(sel_imgs)} -> panel")
#                 print("Prompt (head):", prompt[:120].replace("\n", " / "))
#                 print(f"[tokens] prompt={n_tok}  budget={ctx_budget}")

#                 focus_idx = extract_focus_index(q, max_n=len(sel_imgs))
#                 infer_img = sel_imgs[focus_idx] if focus_idx is not None else panel

#                 bad_turn_ids = make_turn_bad_words_ids(tok, n_imgs=len(sel_imgs))

#                 # 统一准备输入
#                 inputs = vlm.prepare_inputs(infer_img, prompt, ctx_budget)

#                 # 移到设备
#                 for k in list(inputs.keys()):
#                     if isinstance(k, str) and isinstance(inputs[k], torch.Tensor):
#                         inputs[k] = inputs[k].to(vlm.model.device)

#                 # ---- 针对 v1.6：不加禁词；其余模型按原逻辑 ----
#                 gen_kwargs_curr = dict(gen_kwargs)

#                 # --- BLIP-2 专用：禁用 bad_words_ids，并用轻采样避免只输出一个点 ---
#                 if getattr(vlm, "kind", "") and "blip2" in vlm.kind.lower():
#                     # 不加 bad_words_ids
#                     # 轻量确定性采样配置（不影响其它模型）
#                     gen_kwargs_curr.update(dict(
#                         do_sample=True,
#                         temperature=0.3,
#                         top_p=0.92,
#                         num_beams=1,
#                         no_repeat_ngram_size=2,
#                         repetition_penalty=1.05,
#                         use_cache=True,
#                     ))
#                 else:
#                     if bad_turn_ids is not None:
#                         gen_kwargs_curr["bad_words_ids"] = bad_turn_ids

#                 # ---- v1.6 原样输入；其他模型走 pack ----
#                 if getattr(vlm, "kind", "").lower() == "llava_v16":
#                     model_inputs = inputs
#                 else:
#                     model_inputs = _pack_basic_inputs(inputs, vlm.model)

#                 # 稳定生成
#                 output_ids = safe_generate(
#                     vlm.model, model_inputs, gen_kwargs_curr, debug=getattr(args, "debug", False)
#                 )

#                 # ==== 解析输出 ====
#                 # 默认逻辑
#                 inp_len = inputs.get("input_ids", None)
#                 if inp_len is not None and isinstance(inp_len, torch.Tensor):
#                     new_len = output_ids.shape[-1] - inp_len.shape[-1]
#                     gen_start = int(inp_len.shape[-1])
#                 else:
#                     new_len = output_ids.shape[-1]
#                     gen_start = 0

#                 # --- BLIP-2 专用：若只有一个符号，回退到 raw_text ---
#                 ended_with_eos = (eos_id is not None) and (output_ids[0, -1].item() == eos_id)
#                 hit_limit = new_len >= (args.max_new_tokens - 1)

#                 gen_tokens = output_ids[0, gen_start:]  # 只看新增
#                 raw_text = tok.decode(gen_tokens, skip_special_tokens=True)

#                 # 轻量去回声
#                 raw_text = re.sub(r"^\s*(system|user|assistant)\s*[:：]\s*", "", raw_text, flags=re.I)
#                 raw_text = raw_text.replace("You are a helpful assistant.", "").strip()

#                 answer = clean_answer(raw_text, prompt)

#                 # --- 仅对 BLIP-2 启用回退，避免只剩一个 '.' ---
#                 if getattr(vlm, "kind", "") and "blip2" in vlm.kind.lower():
#                     if len(answer.strip()) <= 1:
#                         answer = raw_text.strip()

#                 if hit_limit and not ended_with_eos:
#                     answer = answer.rstrip() + " [TRUNCATED?]"

#                 print("Answer (head):", (answer[:140] + ("..." if len(answer) > 140 else "")))
#                 writer.writerow([i, t+1, idxes_str, files_str, prompt, answer])

#                 # ---------- 每轮结束：主动清理显存 ----------
#                 try:
#                     del output_ids, inputs, model_inputs, infer_img, bad_turn_ids
#                 except Exception:
#                     pass
#                 if torch.cuda.is_available():
#                     torch.cuda.synchronize()
#                     import gc; gc.collect()
#                     torch.cuda.empty_cache()
#                 else:
#                     import gc; gc.collect()

#     print(f"\n✅ Results saved to {args.results}")
#     print("Tip: If you still hit context budget, reduce --max-images-per-turn or --panel-tile-size or --hist-max.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    help="HF model id, e.g., Salesforce/blip2-opt-6.7b | llava-hf/llava-v1.6-vicuna-7b-hf | Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--results", type=str, default="pilot_multimodel_results.csv",
                    help="(legacy) not used for per-sample CSV anymore; kept only for compatibility in printouts")
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--use-history", dest="use_history", action="store_true")
    ap.add_argument("--no-use-history", dest="use_history", action="store_false")
    ap.set_defaults(use_history=True)
    ap.add_argument("--hist-max", type=int, default=2)

    ap.add_argument("--max-images-per-turn", type=int, default=5)
    ap.add_argument("--panel-tile-size", type=int, default=384)
    ap.add_argument("--panel-cols", type=int, default=3)

    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--beams", type=int, default=1)

    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device-map", type=str, default="auto")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--start-sample", type=int, default=0,
                    help="从第几个样本开始（0-based）")

    args = ap.parse_args()

    # ---------- Dataset ----------
    local_root = ensure_mmdu_pics_local()
    print("Local images root:", local_root)

    ds_full = load_dataset(REPO_ID, split="train")

    start = max(0, args.start_sample)
    end = start + args.num_samples if args.num_samples is not None else len(ds_full)
    end = min(end, len(ds_full))

    ds = ds_full.select(range(start, end))
    ds = ds.map(lambda ex: {"image_local": map_to_local_paths(ex["image"], local_root)})

    print(f"[dataset] using samples [{start}, {end}) -> total {len(ds)}")

    # ---------- dtype ----------
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    print(f"Using device_map={args.device_map}, dtype={torch_dtype}")

    # ---------- Load multimodal model ----------
    vlm = load_multimodel(args.model, dtype=torch_dtype, device_map=args.device_map)
    tok = vlm.tokenizer
    eos_id = getattr(tok, "eos_token_id", None)
    pad_id = getattr(tok, "pad_token_id", None)

    # ---------- 显存稳态 ----------
    torch.set_grad_enabled(False)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    # ---------- 生成参数（单处定义） ----------
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
    )
    if eos_id is not None:
        gen_kwargs["eos_token_id"] = eos_id
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id

    if args.do_sample:
        gen_kwargs.update(dict(
            do_sample=True, temperature=args.temperature, top_p=args.top_p,
            num_beams=max(1, args.beams),
        ))
    else:
        gen_kwargs.update(dict(
            do_sample=False,
            num_beams=1,  # 避免 OOM
        ))

    # ---------- context budget ----------
    try:
        cfg = getattr(getattr(vlm.model, "language_model", vlm.model), "config", None)
        max_pos = getattr(cfg, "max_position_embeddings", 2048) if cfg else 2048
    except Exception:
        max_pos = 2048
    margin = 8
    ctx_budget = max(max_pos - args.max_new_tokens - margin, 256)
    print(f"[ctx] max_pos={max_pos}  max_new={args.max_new_tokens}  budget={ctx_budget}")

    # ---------- output dir ----------
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    # 用模型名生成一个短前缀，避免太长/有斜杠
    # 规则：取最后一段（按 "/" split）作为前缀
    model_short = args.model.split("/")[-1]

    # ---------- LOOP OVER SAMPLES ----------
    for i in range(len(ds)):
        ex = ds[i]
        conv = ex["conversations"]
        paths: List[str] = ex["image_local"]

        global_sample_id = start + i  # 这个才是真正的dataset索引

        print("\n" + "=" * 60)
        print(f"Sample {global_sample_id} (local idx {i}) | #images={len(paths)}  #turns={len(conv)}")

        # 为该 sample 打开它自己的 csv
        out_csv_path = os.path.join(
            out_dir,
            f"{model_short}_sample{global_sample_id}.csv"
        )
        print(f"[write] {out_csv_path}")

        with open(out_csv_path, "w", newline="", encoding="utf-8") as f_single:
            writer = csv.writer(f_single)
            # 现在我们写更丰富的列名（跟分析脚本需要的一致性会更好）
            writer.writerow([
                "sample_id", "turn",
                "img_idxes", "image_files",
                "question", "answer"
            ])

            # 预取图像
            sel_paths = paths[: args.max_images_per_turn]
            sel_imgs = [safe_open_image(p) for p in sel_paths]
            short_names = [shorten_name(p) for p in sel_paths]
            idxes_str = ";".join(str(k) for k in range(len(sel_paths)))
            files_str = ";".join(sel_paths)

            # 拼 panel
            panel = make_image_panel(
                imgs=sel_imgs,
                labels=[f"Image{k+1}" for k in range(len(sel_imgs))],
                tile=args.panel_tile_size,
                cols=args.panel_cols,
            )

            # 抽 user 的所有问题
            user_utts = [t["value"] for t in conv if t.get("from") == "user"]

            # 每个样本的实际可跑轮数：min(用户轮数, --max-turns)
            max_turns = min(args.max_turns, len(user_utts))

            # 这个我们现在还没真正用，但保留
            history_summaries: List[str] = []

            for t in range(max_turns):
                q = user_utts[t]

                prompt = build_prompt_multimg(
                    short_names=short_names,
                    recent_ctx=(history_summaries if args.use_history else []),
                    question=q,
                    keep_ctx=args.hist_max,
                )

                # --- BLIP-2 专用提示词简化 ---
                if getattr(vlm, "kind", "") and "blip2" in vlm.kind.lower():
                    prompt = (
                        f"You are given {len(short_names)} images as one panel. "
                        "When I say ImageK, I mean the K-th tile in row-major order (left→right, top→bottom). "
                        "Answer in one concise sentence.\n"
                        f"Q: {q}\nA:"
                    )

                n_tok = vlm.tokenize_len(prompt)
                print(f"\n[Turn {t+1}] | images={len(sel_imgs)} -> panel")
                print("Prompt (head):", prompt[:120].replace("\n", " / "))
                print(f"[tokens] prompt={n_tok}  budget={ctx_budget}")

                # 找到底应该看哪张图（如果问题里提了 Image3 之类）
                focus_idx = extract_focus_index(q, max_n=len(sel_imgs))
                infer_img = sel_imgs[focus_idx] if focus_idx is not None else panel

                bad_turn_ids = make_turn_bad_words_ids(tok, n_imgs=len(sel_imgs))

                # 准备模型输入
                inputs = vlm.prepare_inputs(infer_img, prompt, ctx_budget)

                # tensor移到模型device
                for k_in in list(inputs.keys()):
                    if isinstance(k_in, str) and isinstance(inputs[k_in], torch.Tensor):
                        inputs[k_in] = inputs[k_in].to(vlm.model.device)

                # ---- 针对 v1.6：不加禁词；其他模型按原逻辑 ----
                gen_kwargs_curr = dict(gen_kwargs)

                # --- BLIP-2 特殊采样&不加bad_words_ids ---
                if getattr(vlm, "kind", "") and "blip2" in vlm.kind.lower():
                    gen_kwargs_curr.update(dict(
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.92,
                        num_beams=1,
                        no_repeat_ngram_size=2,
                        repetition_penalty=1.05,
                        use_cache=True,
                    ))
                else:
                    if bad_turn_ids is not None and getattr(vlm, "kind", "").lower() != "llava_v16":
                        gen_kwargs_curr["bad_words_ids"] = bad_turn_ids
                    # 对 llava_v16 我们之前是不要 bad_words_ids 的，所以上面判断一下 kind

                # ---- LLaVA v1.6 特殊：直接喂inputs；否则 pack_basic_inputs ----
                if getattr(vlm, "kind", "").lower() == "llava_v16":
                    model_inputs = inputs
                else:
                    model_inputs = _pack_basic_inputs(inputs, vlm.model)

                # --- 推理 ---
                output_ids = safe_generate(
                    vlm.model, model_inputs, gen_kwargs_curr, debug=getattr(args, "debug", False)
                )

                # ==== 解码 ====
                inp_len_t = inputs.get("input_ids", None)
                if inp_len_t is not None and isinstance(inp_len_t, torch.Tensor):
                    gen_start = int(inp_len_t.shape[-1])
                else:
                    gen_start = 0

                gen_tokens = output_ids[0, gen_start:]
                raw_text = tok.decode(gen_tokens, skip_special_tokens=True)

                # 清理角色前缀
                raw_text = re.sub(r"^\s*(system|user|assistant)\s*[:：]\s*", "",
                                  raw_text, flags=re.I)
                raw_text = raw_text.replace("You are a helpful assistant.", "").strip()

                answer = clean_answer(raw_text, prompt)

                # BLIP-2防止只输出一个符号
                if getattr(vlm, "kind", "") and "blip2" in vlm.kind.lower():
                    if len(answer.strip()) <= 1:
                        answer = raw_text.strip()

                # NOTE: 我们这里保留原始逻辑里 hit_limit -> "[TRUNCATED?]" 的自动标记
                #       方便后续分析截断现象
                ended_with_eos = (eos_id is not None) and (
                    output_ids[0, -1].item() == eos_id
                )
                if inp_len_t is not None and isinstance(inp_len_t, torch.Tensor):
                    new_len = output_ids.shape[-1] - inp_len_t.shape[-1]
                else:
                    new_len = output_ids.shape[-1]
                hit_limit = new_len >= (args.max_new_tokens - 1)

                if hit_limit and not ended_with_eos:
                    answer = answer.rstrip() + " [TRUNCATED?]"

                print("Answer (head):", (answer[:140] + ("..." if len(answer) > 140 else "")))

                # === 写入该 sample 的csv ===
                writer.writerow([
                    global_sample_id,
                    t + 1,
                    idxes_str,
                    files_str,
                    q,
                    answer
                ])

                # --- 回收显存 ---
                try:
                    del output_ids, inputs, model_inputs, infer_img, bad_turn_ids
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    import gc; gc.collect()
                    torch.cuda.empty_cache()
                else:
                    import gc; gc.collect()

    print(f"\n✅ Per-sample CSVs saved under {out_dir}/")
    print("Tip: Each file name is <model_short>_sample<global_id>.csv .")
    print("Tip: If you still hit context budget, reduce --max-images-per-turn or --panel-tile-size or --hist-max.")


if __name__ == "__main__":
    main()
