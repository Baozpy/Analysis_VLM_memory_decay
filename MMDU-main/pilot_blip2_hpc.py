# pilot_blip2_hpc.py — BLIP-2 (HPC/GPU), large models, pixel cache, echo cleaning
import os, zipfile, csv, argparse, torch
from typing import List, Dict, Any
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, Blip2ForConditionalGeneration

# ------------------------------ Defaults ------------------------------
REPO_ID = "laolao77/MMDU"
ZIP_NAME = "mmdu_pics.zip"
EXTRACT_DIR = "./_mmdu_pics"
DEFAULT_MODEL = "Salesforce/blip2-opt-6.7b"      # 可改: Salesforce/blip2-opt-12b
DEFAULT_RESULTS = "pilot_results_hpc.csv"
DEFAULT_MAX_TURNS = 4
DEFAULT_MAX_NEW_TOKENS = 48
DEFAULT_HIST_MAX = 2
DEFAULT_NUM_SAMPLES = 8

# ------------------------------ Utils ------------------------------
def ensure_mmdu_pics_local() -> str:
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    zip_path = hf_hub_download(repo_id=REPO_ID, filename=ZIP_NAME, repo_type="dataset")
    with zipfile.ZipFile(zip_path, "r") as zf:
        if not any(n.lower().endswith((".jpg", ".png", ".jpeg")) for n in zf.namelist()):
            raise RuntimeError("mmdu_pics.zip 中未发现图像文件")
        zf.extractall(EXTRACT_DIR)
    root = os.path.join(EXTRACT_DIR, "mmdu_pics")
    return os.path.abspath(root if os.path.isdir(root) else EXTRACT_DIR)

def map_to_local_paths(paths: List[str], local_root: str) -> List[str]:
    mapped = []
    for p in paths:
        sub = p.lstrip("/")
        rel = os.path.relpath(sub, "mmdu_pics")
        mapped.append(os.path.join(local_root, rel))
    return mapped

def build_prompt(history: List[str], question: str, max_prev: int) -> str:
    def brief(q, limit=80):
        q = q.replace("\n", " ").strip()
        return (q[:limit] + "…") if len(q) > limit else q
    prev = " ".join([f"[Prev] {brief(h)}" for h in history[-max_prev:]])
    if prev:
        return f"{prev}\nQuestion: {question}\nAnswer in one short sentence. Do not repeat the question. Answer:"
    return f"Question: {question}\nAnswer in one short sentence. Do not repeat the question. Answer:"

def clean_answer(raw: str) -> str:
    txt = raw.strip()
    if "Answer:" in txt:
        txt = txt.split("Answer:")[-1].strip()
    lines = [
        ln for ln in txt.splitlines()
        if not ln.strip().startswith(("Question:", "Previously asked:", "[Prev]"))
    ]
    txt = " ".join(ln.strip() for ln in lines).strip()
    if not txt:
        txt = raw.replace("Question:", "").replace("Answer:", "").strip()
    return txt

def get_tok_ids(processor, model):
    eos_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
    pad_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
    if eos_id is None:
        eos_id = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if pad_id is None:
        pad_id = eos_id
    return eos_id, pad_id

def load_image_pil(path: str):
    from PIL import Image as PILImage
    return PILImage.open(path).convert("RGB")

# ------------------------------ Main ------------------------------
def main():
    parser = argparse.ArgumentParser(description="BLIP-2 Large (HPC/GPU) pilot")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--results", type=str, default=DEFAULT_RESULTS)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--hist-max", type=int, default=DEFAULT_HIST_MAX)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16","fp16","fp32"],
                        help="model dtype (bf16 recommended on A100/H100)")
    parser.add_argument("--device-map", type=str, default="auto",
                        help='set to "auto" for multi-GPU sharding; or "cuda" for single GPU')
    parser.add_argument("--load-in-8bit", action="store_true", help="quantize to 8-bit with bitsandbytes")
    parser.add_argument("--load-in-4bit", action="store_true", help="quantize to 4-bit with bitsandbytes")
    parser.add_argument("--no-pixel-cache", action="store_true", help="disable pixel_values caching")
    args = parser.parse_args()

    # 环境优化
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    local_root = ensure_mmdu_pics_local()
    print("Local images root:", local_root)

    ds = load_dataset(REPO_ID, split="train").select(range(args.num_samples))
    ds = ds.map(lambda ex: {"image_local": map_to_local_paths(ex["image"], local_root)})

    # 选择 dtype
    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Processor & Model
    processor = AutoProcessor.from_pretrained(args.model)

    model_kwargs: Dict[str, Any] = {
        "device_map": args.device_map,
    }
    # 量化选项（节省显存）
    if args.load_in_4bit:
        model_kwargs.update(dict(load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype))
    elif args.load_in_8bit:
        model_kwargs.update(dict(load_in_8bit=True, torch_dtype=torch_dtype))
    else:
        # 非量化：直接设置 dtype
        # 某些 transformers 版本用 torch_dtype，新版也支持 dtype
        try:
            model_kwargs.update(dict(dtype=torch_dtype))
        except TypeError:
            model_kwargs.update(dict(torch_dtype=torch_dtype))

    print(f"Loading model: {args.model}")
    model = Blip2ForConditionalGeneration.from_pretrained(args.model, **model_kwargs)
    model.eval()

    eos_id, pad_id = get_tok_ids(processor, model)
    print(f"EOS={eos_id}, PAD={pad_id}, dtype={torch_dtype}, device_map={args.device_map}")

    # 像素缓存（同一张图多回合复用）
    pixel_cache: Dict[str, torch.Tensor] = {}

    with open(args.results, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "turn", "img_idx", "prompt", "answer"])

        for i in range(len(ds)):
            ex = ds[i]
            conv = ex["conversations"]
            images = ex["image_local"]
            print("\n" + "="*70)
            print(f"Sample {i} | #images={len(images)}  #turns={len(conv)}")

            user_utts = [t["value"] for t in conv if t["from"] == "user"]
            turns = min(args.max_turns, len(user_utts))
            history: List[str] = []

            for t in range(turns):
                q = user_utts[t]
                img_idx = min(t, len(images) - 1)
                img_path = images[img_idx]

                # 1) pixel_values 缓存
                if (not args.no_pixel_cache) and img_path in pixel_cache:
                    pixel_values = pixel_cache[img_path]
                else:
                    pil = load_image_pil(img_path)
                    pixel_values = processor(images=pil, return_tensors="pt")["pixel_values"]
                    # 把 pixel_values 自动放置到模型第一块设备上
                    dev0 = next(model.parameters()).device
                    pixel_values = pixel_values.to(dev0)
                    if not args.no_pixel_cache:
                        pixel_cache[img_path] = pixel_values

                # 2) 文本处理（很轻）
                prompt = build_prompt(history, q, args.hist_max)
                text_inputs = processor(text=prompt, return_tensors="pt")
                for k in text_inputs:
                    text_inputs[k] = text_inputs[k].to(pixel_values.device)

                # 3) 生成
                with torch.inference_mode():
                    out = model.generate(
                        pixel_values=pixel_values,
                        input_ids=text_inputs["input_ids"],
                        attention_mask=text_inputs.get("attention_mask", None),
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                        eos_token_id=eos_id,
                        pad_token_id=pad_id,
                    )
                raw = processor.batch_decode(out, skip_special_tokens=True)[0]
                ans = clean_answer(raw)

                print(f"[Turn {t+1}] img_idx={img_idx}")
                print("Prompt:", prompt[:140].replace("\n"," / "))
                print("Raw   :", raw[:140].replace("\n"," / "))
                print("Answer:", ans)

                writer.writerow([i, t+1, img_idx, prompt, ans])
                history.append(q)

    print(f"\n✅ Results saved to {args.results}")
    print("Done.")

if __name__ == "__main__":
    main()
