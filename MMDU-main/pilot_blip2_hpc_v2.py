# pilot_blip2_hpc_v2.py (fixed dtype handling + decoding args)
import os, csv, argparse, zipfile
from typing import List
from datasets import load_dataset, Image as HFImage, Sequence
from huggingface_hub import hf_hub_download
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image as PILImage

REPO_ID = "laolao77/MMDU"
ZIP_NAME = "mmdu_pics.zip"
EXTRACT_DIR = "./_mmdu_pics"

def ensure_mmdu_pics_local() -> str:
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    zip_path = hf_hub_download(repo_id=REPO_ID, filename=ZIP_NAME, repo_type="dataset")
    with zipfile.ZipFile(zip_path, "r") as zf:
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

def build_prompt(history, new_q, max_prev=2):
    ctx = " ".join(f"Context: {h}" for h in history[-max_prev:])
    q = f"Question: {new_q} Short answer:"
    return (ctx + " " + q).strip() if ctx else q

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Salesforce/blip2-opt-6.7b")
    parser.add_argument("--results", type=str, default="pilot_results_v2.csv")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--hist-max", type=int, default=2)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", type=str, default="auto")
    # 解码与防复读
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling (needed if using temperature/top_p).")
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    args = parser.parse_args()

    local_root = ensure_mmdu_pics_local()
    print("Local images root:", local_root)

    raw = load_dataset(REPO_ID, split="train").select(range(args.num_samples))
    raw = raw.map(lambda ex: {
        "image_path_list": map_to_local_paths(ex["image"], local_root),
        "image_local": map_to_local_paths(ex["image"], local_root),
    })
    ds = raw.cast_column("image_local", Sequence(HFImage()))
    print("Dataset ready.")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    print(f"Using device_map={args.device_map}, dtype={torch_dtype}")

    processor = Blip2Processor.from_pretrained(args.model)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model,
        device_map=args.device_map,
        torch_dtype=torch_dtype,  # 只影响浮点权重
    )

    eos_token_id = processor.tokenizer.eos_token_id
    pad_token_id = processor.tokenizer.pad_token_id

    with open(args.results, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "turn", "img_idx", "image_path", "prompt", "answer"])

        for i in range(len(ds)):
            ex = ds[i]
            conv = ex["conversations"]
            image_paths = ex["image_path_list"]

            print("\n" + "="*60)
            print(f"Sample {i} | #images={len(image_paths)}  #turns={len(conv)}")

            user_utts = [t["value"] for t in conv if t["from"] == "user"]
            max_turns = min(args.max_turns, len(user_utts))

            for t in range(max_turns):
                question = user_utts[t]
                img_idx = min(t, len(image_paths) - 1)
                image_path = image_paths[img_idx]
                pil_img = PILImage.open(image_path).convert("RGB")

                prompt = build_prompt(user_utts[:t], question, args.hist_max)

                # 只把张量搬到 device；保持 input_ids/attention_mask 的整型，不改变 dtype
                inputs = processor(images=pil_img, text=prompt, return_tensors="pt")
                for k, v in inputs.items():
                    if hasattr(v, "to"):
                        inputs[k] = v.to(model.device)
                # 仅将像素张量转成目标浮点精度
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch_dtype)

                gen_kwargs = dict(
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    num_beams=args.beams,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                )
                # 采样配置
                if args.do_sample or args.temperature != 1.0 or args.top_p != 1.0:
                    gen_kwargs.update(dict(
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    ))

                output_ids = model.generate(**inputs, **gen_kwargs)
                answer = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

                # 简单“可能被截断”的提示
                if "input_ids" in inputs and len(output_ids[0]) >= inputs["input_ids"].shape[1] + args.max_new_tokens - 1:
                    answer += " [TRUNCATED?]"

                print(f"\n[Turn {t+1}] img_idx={img_idx} | image_path={image_path}")
                print("Prompt:", prompt[:140].replace("\n", " / "))
                print("Answer:", (answer[:220] + ("..." if len(answer) > 220 else "")))

                writer.writerow([i, t+1, img_idx, image_path, prompt, answer])

    print(f"\n✅ Results saved to {args.results}")

if __name__ == "__main__":
    main()
