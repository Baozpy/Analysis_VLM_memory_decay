#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from PIL import Image

from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab
from skimage.feature import canny
from scipy.ndimage import binary_dilation

import torch
import torchvision.transforms as T

_QWEN_SINGLETON = {"model": None, "processor": None, "device": None, "model_id": None}


# Optional deps
try:
    import lpips  # type: ignore
except Exception:  # pragma: no cover
    lpips = None  # type: ignore

try:
    import timm  # type: ignore
except Exception:  # pragma: no cover
    timm = None  # type: ignore

try:
    from transformers import (
        AutoImageProcessor, AutoModel, AutoTokenizer, AutoProcessor,
        AutoModelForCausalLM, AutoModelForVision2Seq  # type: ignore
    )
except Exception:  # pragma: no cover
    AutoImageProcessor = None  # type: ignore
    AutoModel = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    AutoProcessor = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoModelForVision2Seq = None  # type: ignore

# Hardcoded model IDs (保持与原脚本一致)
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DINO_MODEL_ID = "facebook/dinov2-base"


@dataclass
class CompareConfig:
    image_path_a: str
    image_path_b: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_net: str = "alex"
    canny_sigma: float = 1.0
    edge_match_radius: int = 1
    max_edge_side: int = 512


def load_image_rgb(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def to_numpy_float01(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def pil_resize(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return img.resize(size, Image.BICUBIC)


def compute_ssim_metric(img_a: Image.Image, img_b: Image.Image) -> float:
    if img_a.size != img_b.size:
        img_b = pil_resize(img_b, img_a.size)
    a = to_numpy_float01(img_a)
    b = to_numpy_float01(img_b)
    value = ssim(a, b, channel_axis=-1, data_range=1.0)
    return float(value)


def compute_lpips_metric(img_a: Image.Image, img_b: Image.Image, device: str, net: str = "alex") -> Optional[float]:
    if lpips is None:
        return None
    model = lpips.LPIPS(net=net).to(device)
    model.eval()

    def to_lpips_tensors(img_a: Image.Image, img_b: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        if img_a.size != img_b.size:
            img_b = pil_resize(img_b, img_a.size)
        ta = T.ToTensor()(img_a).unsqueeze(0)
        tb = T.ToTensor()(img_b).unsqueeze(0)
        ta = ta * 2.0 - 1.0
        tb = tb * 2.0 - 1.0
        return ta.to(device), tb.to(device)

    with torch.no_grad():
        ta, tb = to_lpips_tensors(img_a, img_b)
        dist = model(ta, tb)
        return float(dist.item())


def compute_mae_delta(img_a: Image.Image, img_b: Image.Image) -> float:
    if img_a.size != img_b.size:
        img_b = pil_resize(img_b, img_a.size)
    a = to_numpy_float01(img_a)
    b = to_numpy_float01(img_b)
    return float(np.mean(np.abs(a - b)))


def compute_ciede2000(img_a: Image.Image, img_b: Image.Image) -> Tuple[float, float]:
    if img_a.size != img_b.size:
        img_b = pil_resize(img_b, img_a.size)
    a = to_numpy_float01(img_a)
    b = to_numpy_float01(img_b)
    lab_a = rgb2lab(a)
    lab_b = rgb2lab(b)

    L1, a1, b1 = lab_a[..., 0], lab_a[..., 1], lab_a[..., 2]
    L2, a2, b2 = lab_b[..., 0], lab_b[..., 1], lab_b[..., 2]

    kL = 1.0
    kC = 1.0
    kH = 1.0

    C1 = np.sqrt(a1 * a1 + b1 * b1)
    C2 = np.sqrt(a2 * a2 + b2 * b2)
    C_bar = 0.5 * (C1 + C2)
    C_bar7 = C_bar ** 7
    G = 0.5 * (1 - np.sqrt(C_bar7 / (C_bar7 + 25 ** 7)))

    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p * a1p + b1 * b1)
    C2p = np.sqrt(a2p * a2p + b2 * b2)

    def atan2d(y, x):
        ang = np.degrees(np.arctan2(y, x))
        ang[ang < 0] += 360
        return ang

    h1p = atan2d(b1, a1p)
    h2p = atan2d(b2, a2p)

    dLp = L2 - L1
    dCp = C2p - C1p

    def dh(h1, h2):
        diff = h2 - h1
        diff[diff > 180] -= 360
        diff[diff < -180] += 360
        return diff

    dhp = dh(h1p, h2p)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    Lp_bar = 0.5 * (L1 + L2)
    Cp_bar = 0.5 * (C1p + C2p)

    def h_bar(h1, h2):
        hb = (h1 + h2) / 2.0
        mask = np.abs(h1 - h2) > 180
        hb[mask] += 180
        hb[hb >= 360] -= 360
        return hb

    hp_bar = h_bar(h1p, h2p)

    T = (
        1
        - 0.17 * np.cos(np.radians(hp_bar - 30))
        + 0.24 * np.cos(np.radians(2 * hp_bar))
        + 0.32 * np.cos(np.radians(3 * hp_bar + 6))
        - 0.20 * np.cos(np.radians(4 * hp_bar - 63))
    )

    dtheta = 30 * np.exp(-((hp_bar - 275) / 25) ** 2)
    RC = 2 * np.sqrt((Cp_bar ** 7) / (Cp_bar ** 7 + 25 ** 7))
    SL = 1 + (0.015 * (Lp_bar - 50) ** 2) / np.sqrt(20 + (Lp_bar - 50) ** 2)
    SC = 1 + 0.045 * Cp_bar
    SH = 1 + 0.015 * Cp_bar * T
    RT = -np.sin(np.radians(2 * dtheta)) * RC

    dE = np.sqrt(
        (dLp / (kL * SL)) ** 2
        + (dCp / (kC * SC)) ** 2
        + (dHp / (kH * SH)) ** 2
        + RT * (dCp / (kC * SC)) * (dHp / (kH * SH))
    )

    return float(np.mean(dE)), float(np.median(dE))


def compute_edge_f1(img_a: Image.Image, img_b: Image.Image, sigma: float = 1.0, match_radius: int = 1, max_side: int = 512) -> float:
    def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
        w, h = img.size
        scale = max_side / max(w, h)
        if scale >= 1.0:
            return img
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return pil_resize(img, (new_w, new_h))

    A = resize_max_side(img_a, max_side)
    B = resize_max_side(img_b, max_side)
    if A.size != B.size:
        B = pil_resize(B, A.size)

    a_gray = np.asarray(A.convert("L"), dtype=np.float32) / 255.0
    b_gray = np.asarray(B.convert("L"), dtype=np.float32) / 255.0

    edges_a = canny(a_gray, sigma=sigma)
    edges_b = canny(b_gray, sigma=sigma)

    if match_radius > 0:
        structure = np.ones((2 * match_radius + 1, 2 * match_radius + 1), dtype=bool)
        edges_a_d = binary_dilation(edges_a, structure)
        edges_b_d = binary_dilation(edges_b, structure)
    else:
        edges_a_d, edges_b_d = edges_a, edges_b

    tp = np.logical_and(edges_a, edges_b_d).sum()
    fp = np.logical_and(edges_a, np.logical_not(edges_b_d)).sum()
    fn = np.logical_and(np.logical_not(edges_a), edges_b).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(f1)




def _resize_for_vlm(pil_img, max_side=448):
    # 等比缩放到最长边 = max_side（PIL 原地改）
    img = pil_img.copy()
    w, h = img.size
    scale = max_side / float(max(w, h))
    if scale < 1.0:
        img = img.resize((int(round(w*scale)), int(round(h*scale))), Image.BICUBIC)
    return img

def qwen_simple_prompt(images: list, prompt: str, model_id: str, device: str, verbose: bool=False):
    if AutoProcessor is None or AutoModelForVision2Seq is None:
        return None, ("error: transformers AutoProcessor/AutoModelForVision2Seq not available; "
                      "pip install -U transformers accelerate safetensors")

    use_cuda = (torch.cuda.is_available() and device.startswith("cuda"))
    dtype = torch.bfloat16 if use_cuda else torch.float32

    cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE", None)
    local_only = os.environ.get("HF_LOCAL_ONLY", "0") == "1"

    # ---------- 单例缓存 ----------
    global _QWEN_SINGLETON
    must_reload = (
        _QWEN_SINGLETON["model"] is None
        or _QWEN_SINGLETON["processor"] is None
        or _QWEN_SINGLETON["device"] != device
        or _QWEN_SINGLETON["model_id"] != model_id
    )

    if must_reload:
        try:
            if use_cuda:
                torch.cuda.empty_cache()  # 加载前清一次
        except Exception:
            pass
        try:
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto" if device.startswith("cuda") else None,
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=local_only,
            )
            if not device.startswith("cuda"):
                model = model.to(device)
            processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True, cache_dir=cache_dir, local_files_only=local_only
            )
            _QWEN_SINGLETON.update({"model": model, "processor": processor, "device": device, "model_id": model_id})
            if verbose:
                print(f"[Qwen] loaded once: model={model_id}, device={device}, dtype={dtype}")
        except Exception as e:
            return None, f"error: load_failed: {type(e).__name__}: {e}"
    else:
        model = _QWEN_SINGLETON["model"]
        processor = _QWEN_SINGLETON["processor"]

    # ---------- 预处理：把两张图等比缩到最长边 448（强力降显存） ----------
    imgs_small = [_resize_for_vlm(img, max_side=448) for img in images]

    try:
        if hasattr(processor, "apply_chat_template"):
            messages = [
                {"role": "user", "content": ([{"type": "image", "image": img} for img in imgs_small] + [{"type": "text", "text": prompt}])}
            ]
            chat_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=[chat_text], images=imgs_small, return_tensors="pt")
        else:
            if verbose:
                print("warning: processor missing apply_chat_template; falling back to plain processor inputs")
            inputs = processor(text=[prompt], images=imgs_small, return_tensors="pt")

        model_device = next(iter(model.parameters())).device
        inputs = {k: (v.to(model_device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        # ---------- 生成前后清显存 ----------
        try:
            if use_cuda:
                torch.cuda.empty_cache()
        except Exception:
            pass

        with torch.no_grad():
            # 关键：缩短输出 + 关闭 KV cache，极大降低峰值
            output_ids = model.generate(**inputs, max_new_tokens=12, do_sample=False, use_cache=False)

        try:
            if use_cuda:
                torch.cuda.empty_cache()
        except Exception:
            pass

        text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        # 稳健解析 1–5（避免命中 “1-5” 的 1）
        import re
        rating = None
        m = re.search(r"\bSCORE\s*:\s*([1-5])\b", text, re.I)
        if not m:
            m = re.search(r"(?:score|rating)\s*[:=]\s*([1-5])", text, re.I)
        if not m:
            m = re.search(r"\bis\s*([1-5])\b", text, re.I)
        if not m:
            digits = re.findall(r"(?<!-)\b([1-5])\b(?!-)", text)
            if digits:
                rating = int(digits[-1])
        else:
            rating = int(m.group(1))

        # 返回文本，调用处把 rating 填入 JSON（或你也可以扩展成一起返回）
        return text, None

    except RuntimeError as e:
        # ---------- OOM 兜底：强制 CPU 重跑一次，保证出分 ----------
        if "out of memory" in str(e).lower() and use_cuda:
            try:
                if verbose:
                    print("[Qwen] GPU OOM; fallback to CPU once.")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                # 卸掉 GPU 模型，防止占显存
                _QWEN_SINGLETON["model"] = None
                _QWEN_SINGLETON["processor"] = None
                _QWEN_SINGLETON["device"] = None
                _QWEN_SINGLETON["model_id"] = None
                # 递归用 CPU 跑一次（会走到上面的加载分支，但放到 CPU）
                return qwen_simple_prompt(images, prompt, model_id, device="cpu", verbose=verbose)
            except Exception as e2:
                return None, f"error: generate_failed_after_cpu_fallback: {type(e2).__name__}: {e2}"
        return None, f"error: generate_failed: {type(e).__name__}: {e}"
    except Exception as e:
        return None, f"error: generate_failed: {type(e).__name__}: {e}"




def _dino_pairwise(img_a: Image.Image, img_b: Image.Image, device: str) -> Optional[Dict[str, float]]:
    if AutoImageProcessor is None or AutoModel is None:
        return None
    try:
        processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
        model = AutoModel.from_pretrained(DINO_MODEL_ID).to(device).eval()
        with torch.no_grad():
            ia = processor(images=img_a, return_tensors="pt").to(device)
            ib = processor(images=img_b, return_tensors="pt").to(device)
            fa = model(**ia).last_hidden_state.mean(dim=1)
            fb = model(**ib).last_hidden_state.mean(dim=1)
            cos = torch.nn.functional.cosine_similarity(fa, fb).item()
            sim01 = (cos + 1.0) / 2.0
            l2 = torch.norm(fa[0] - fb[0], p=2).item()
        return {"cosine_similarity": float(cos), "similarity_01": float(sim01), "l2_distance": float(l2)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def compare_one_pair(path_a: str, path_b: str, device: str, lpips_net: str, use_qwen: bool, use_dino: bool, verbose: bool=False) -> Dict[str, Any]:
    t0 = time.time()
    img_a = load_image_rgb(path_a)
    img_b = load_image_rgb(path_b)

    ssim_v = compute_ssim_metric(img_a, img_b)
    lpips_v = compute_lpips_metric(img_a, img_b, device, lpips_net)
    mae_v = compute_mae_delta(img_a, img_b)
    dE_mean, dE_median = compute_ciede2000(img_a, img_b)
    edge_f1_v = compute_edge_f1(img_a, img_b, sigma=1.0, match_radius=1, max_side=512)

    qwen_raw = None
    qwen_err = None
    qwen_rating = None
    if use_qwen:
        # —— 跑 Qwen 前先把前面步骤（LPIPS/DINO）的显存碎片清掉 ——
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        prompt = (
             "You are a strict image similarity judge for scientific plots and charts. "
             "Given two images, output ONLY ONE line in this exact format:\n"
             "SCORE: <1-5>; REASON: <a single short sentence>"
        )
        qwen_raw, qwen_err = qwen_simple_prompt([img_a, img_b], prompt, QWEN_MODEL_ID, device, verbose=verbose)

        # —— 跑完也清一次，避免后续样本累积碎片 ——
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # 解析 1–5 分（更稳正则；若你已放在 qwen_simple_prompt 里，可删掉这一段）
        if qwen_raw and qwen_rating is None:
            import re
            text = qwen_raw
            m = re.search(r"(?:score|rating)\s*[:=]\s*([1-5])", text, re.I)
            if not m:
                m = re.search(r"\bis\s*([1-5])\b", text, re.I)
            if not m:
                digits = re.findall(r"(?<!-)\b([1-5])\b(?!-)", text)
                if digits:
                    qwen_rating = int(digits[-1])
            else:
                qwen_rating = int(m.group(1))

    
    dino_section = None
    if use_dino:
        dino_section = _dino_pairwise(img_a, img_b, device)
        if dino_section is None and verbose:
            print("DINO requested but transformers AutoImageProcessor/AutoModel not available")

    out: Dict[str, Any] = {
        "image_a": os.path.abspath(path_a),
        "image_b": os.path.abspath(path_b),
        "device": device,
        "metrics": {
            "ssim": ssim_v,
            "lpips": lpips_v,
            "mae_delta": mae_v,
            "ciede2000_mean": dE_mean,
            "ciede2000_median": dE_median,
            "edge_f1": edge_f1_v,
        },
        "qwen": {
            "model": QWEN_MODEL_ID if use_qwen else None,
            "rating_1to5": qwen_rating,
            "raw_response": qwen_raw if qwen_raw is not None else qwen_err,
        } if use_qwen else None,
        "dinov3": dino_section,
        "elapsed_sec": time.time() - t0,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare image pairs with multiple metrics. Single or batch mode.")
    # 原有单对输入（保留默认值）
    parser.add_argument("image_a", type=str, nargs="?", default="/fs/clip-scratch/apalnitk/figure/scripts/plot2code_outputs/sample_0/generated.png", help="Path to first image")
    parser.add_argument("image_b", type=str, nargs="?", default="/fs/clip-scratch/apalnitk/figure/scripts/plot2code_outputs/sample_0/ground_truth.png", help="Path to second image")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lpips-net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--no-qwen", action="store_true", help="Skip Qwen rating")
    parser.add_argument("--no-dino", action="store_true", help="Skip DINO embedding similarity")
    parser.add_argument("--verbose", action="store_true", help="Print detailed errors and debug info")

    # 新增：批量比较
    parser.add_argument("--batch-root", type=str,
                        default="/fs/clip-scratch/apalnitk/figure/scripts/plot2code_outputs",
                        help="Root directory containing sample_*/ subfolders")
    parser.add_argument("--start", type=int, default=None, help="Start index for samples (e.g., 0)")
    parser.add_argument("--end", type=int, default=None, help="End index inclusive (e.g., 367)")
    parser.add_argument("--out-csv", type=str, default=None, help="Write batch results to CSV (id,paths,metrics,VLMS)")

    args = parser.parse_args()

    # 如果指定了 start/end，则进入批量模式；否则走单对比较（保持原行为）
    if args.start is not None and args.end is not None:
        # 批量模式
        rows: List[Dict[str, Any]] = []
        for i in range(args.start, args.end + 1):
            sample_dir = os.path.join(args.batch_root, f"sample_{i}")
            gen = os.path.join(sample_dir, "generated.png")
            gt = os.path.join(sample_dir, "ground_truth.png")
            if not (os.path.exists(gen) and os.path.exists(gt)):
                if args.verbose:
                    print(f"[SKIP] Missing files in {sample_dir}")
                continue
            result = compare_one_pair(
                gen, gt, device=args.device, lpips_net=args.lpips_net,
                use_qwen=(not args.no_qwen), use_dino=(not args.no_dino), verbose=args.verbose
            )
            # 打印逐条 JSON（便于流式查看）
            print(json.dumps({"id": f"sample_{i}", **result}, indent=2))

            # 收集到 CSV 行
            metrics = result.get("metrics", {})
            qwen = result.get("qwen") or {}
            dino = result.get("dinov3") or {}
            rows.append({
                "id": f"sample_{i}",
                "image_a": result["image_a"],
                "image_b": result["image_b"],
                "ssim": metrics.get("ssim"),
                "lpips": metrics.get("lpips"),
                "mae_delta": metrics.get("mae_delta"),
                "ciede2000_mean": metrics.get("ciede2000_mean"),
                "ciede2000_median": metrics.get("ciede2000_median"),
                "edge_f1": metrics.get("edge_f1"),
                "qwen_rating_1to5": qwen.get("rating_1to5") if qwen else None,
                "qwen_raw": qwen.get("raw_response") if qwen else None,
                "dino_cosine": dino.get("cosine_similarity") if isinstance(dino, dict) else None,
                "dino_similarity01": dino.get("similarity_01") if isinstance(dino, dict) else None,
                "dino_l2": dino.get("l2_distance") if isinstance(dino, dict) else None,
                "elapsed_sec": result.get("elapsed_sec"),
            })

        if args.out_csv:
            # 写 CSV
            import pandas as pd
            import pathlib
            import json as pyjson

            # 写 CSV
            pathlib.Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.out_csv, index=False)
            print(f"[OK] Saved batch CSV -> {args.out_csv}")

            # 写 JSON（文件名和 CSV 并排，只是扩展名不同）
            out_json = str(pathlib.Path(args.out_csv).with_suffix(".json"))
            with open(out_json, "w") as f:
                pyjson.dump(rows, f, indent=2, ensure_ascii=False)
            print(f"[OK] Saved batch JSON -> {out_json}")
        return

    # 单对比较（和原脚本一致）
    cfg = CompareConfig(
        image_path_a=args.image_a,
        image_path_b=args.image_b,
        device=args.device,
        lpips_net=args.lpips_net,
    )
    result = compare_one_pair(
        cfg.image_path_a, cfg.image_path_b, device=cfg.device, lpips_net=cfg.lpips_net,
        use_qwen=(not args.no_qwen), use_dino=(not args.no_dino), verbose=args.verbose
    )
    print(json.dumps(result, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
