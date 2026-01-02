# pilot_blip2.py — BLIP-2 VQA (no pipeline, with echo cleaning)
import os, zipfile, csv, argparse, torch
from typing import List
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image as PILImage

# ------------------------------ Defaults ------------------------------
REPO_ID = "laolao77/MMDU"
ZIP_NAME = "mmdu_pics.zip"
EXTRACT_DIR = "./_mmdu_pics"
DEFAULT_MODEL = "Salesforce/blip2-opt-2.7b"      # 公开可用
DEFAULT_RESULTS = "pilot_results.csv"
DEFAULT_MAX_TURNS = 4
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_HIST_MAX = 2
DEFAULT_NUM_SAMPLES = 3

# ------------------------------ Utils ------------------------------
def ensure_mmdu_pics_local() -> str:
    """确保 mmdu_pics.zip 已解压到本地，返回图片根目录绝对路径"""
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    zip_path = hf_hub_download(repo_id=REPO_ID, filename=ZIP_NAME, repo_type="dataset")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # 幂等解压：如果已解压过不重复开销
        if not any(n.lower().endswith((".jpg", ".png", ".jpeg")) for n in zf.namelist()):
            raise RuntimeError("mmdu_pics.zip 中未发现图像文件")
        zf.extractall(EXTRACT_DIR)
    root = os.path.join(EXTRACT_DIR, "mmdu_pics")
    return os.path.abspath(root if os.path.isdir(root) else EXTRACT_DIR)

def map_to_local_paths(paths: List[str], local_root: str) -> List[str]:
    """将数据集里的 '/mmdu_pics/...' 路径映射为当前机器的本地绝对路径"""
    mapped = []
    for p in paths:
        sub = p.lstrip("/")  # 去掉开头的斜杠
        rel = os.path.relpath(sub, "mmdu_pics")
        mapped.append(os.path.join(local_root, rel))
    return mapped

def detect_device_dtype():
    """CUDA→fp16；MPS/CPU→fp32（更稳）"""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32

def build_prompt(history: List[str], question: str, max_prev: int) -> str:
    """精简历史上下文 + 当前问题，减少回声机会"""
    def brief(q, limit=80):
        q = q.replace("\n", " ").strip()
        return (q[:limit] + "…") if len(q) > limit else q
    prev = " ".join([f"[Prev] {brief(h)}" for h in history[-max_prev:]])
    if prev:
        return f"{prev}\nQuestion: {question}\nAnswer concisely in one sentence. Answer:"
    return f"Question: {question}\nAnswer concisely in one sentence. Answer:"

def load_image_flex(entry):
    """兼容 字符串路径 / PIL.Image / dict{'path':...} 三种情况"""
    if isinstance(entry, str):
        return PILImage.open(entry).convert("RGB")
    if hasattr(entry, "convert"):  # PIL.Image.Image
        return entry.convert("RGB")
    if isinstance(entry, dict) and "path" in entry:
        return PILImage.open(entry["path"]).convert("RGB")
    raise ValueError(f"Unexpected image type: {type(entry)}")

def clean_answer(raw: str) -> str:
    """去回声：剔除 Question/Prev 模板，仅保留 Answer 内容"""
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

def get_special_token_ids(processor, model):
    """获取 eos/pad token id（多重兜底）"""
    eos_id = None
    pad_id = None
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        eos_id = getattr(tok, "eos_token_id", None)
        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is None:
            # 有些 tokenizer 无 pad，用 eos 兜底
            pad_id = eos_id
    # 再从模型的 generation_config 兜底
    if eos_id is None:
        eos_id = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if pad_id is None:
        pad_id = eos_id
    return eos_id, pad_id

# ------------------------------ Main ------------------------------
def main():
    parser = argparse.ArgumentParser(description="BLIP-2 VQA pilot (no pipeline)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--results", type=str, default=DEFAULT_RESULTS)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--hist-max", type=int, default=DEFAULT_HIST_MAX)
    args = parser.parse_args()

    local_root = ensure_mmdu_pics_local()
    print("Local images root:", local_root)

    # 取前 N 个样本做试跑
    ds = load_dataset(REPO_ID, split="train").select(range(args.num_samples))
    ds = ds.map(lambda ex: {"image_local": map_to_local_paths(ex["image"], local_root)})

    device, dtype = detect_device_dtype()
    print(f"Using device: {device}, dtype: {dtype}")

    processor = AutoProcessor.from_pretrained(args.model)
    # transformers >= 4.44 用 dtype；低版本用 torch_dtype 也能兼容
    try:
        model = Blip2ForConditionalGeneration.from_pretrained(args.model, dtype=dtype)
    except TypeError:
        model = Blip2ForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    eos_id, pad_id = get_special_token_ids(processor, model)

    with open(args.results, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "turn", "img_idx", "prompt", "answer"])

        for i in range(len(ds)):
            ex = ds[i]
            conv = ex["conversations"]
            images = ex["image_local"]
            print("\n" + "="*60)
            print(f"Sample {i} | #images={len(images)}  #turns={len(conv)}")

            user_utts = [t["value"] for t in conv if t["from"] == "user"]
            turns = min(args.max_turns, len(user_utts))
            history: List[str] = []

            for t in range(turns):
                q = user_utts[t]
                img_idx = min(t, len(images) - 1)
                image = load_image_flex(images[img_idx])

                prompt = build_prompt(history, q, args.hist_max)
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,              # 稳定可复现；要多样性可设 True
                        num_beams=1,                  # 可调为 3 提升稳健性
                        repetition_penalty=1.2,       # 抑制回声
                        no_repeat_ngram_size=3,       # 禁止 3-gram 重复
                        eos_token_id=eos_id,
                        pad_token_id=pad_id,
                        early_stopping=True,
                    )
                raw = processor.batch_decode(out, skip_special_tokens=True)[0]
                ans = clean_answer(raw)

                print(f"\n[Turn {t+1}] img_idx={img_idx}")
                print("Prompt:", prompt[:140].replace("\n"," / "))
                print("Raw:", raw[:140].replace("\n"," / "))
                print("Answer:", ans)

                writer.writerow([i, t+1, img_idx, prompt, ans])
                history.append(q)

    print(f"\n✅ Results saved to {args.results}")

if __name__ == "__main__":
    main()
