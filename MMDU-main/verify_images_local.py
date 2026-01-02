# save as: verify_images_local.py
import os, zipfile, pathlib
from huggingface_hub import hf_hub_download
from datasets import load_dataset, Image, Sequence

REPO_ID = "laolao77/MMDU"
ZIP_NAME = "mmdu_pics.zip"   # 如果后续需要更大集，可换成 "mmdu-45k_pics.zip"
EXTRACT_DIR = "./_mmdu_pics" # 解压到当前目录下

def ensure_mmdu_pics_local():
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    # 1) 下载 zip（带缓存）
    zip_path = hf_hub_download(repo_id=REPO_ID, filename=ZIP_NAME, repo_type="dataset")
    # 2) 解压（若尚未解压或不完整）
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # 判断是否已经解压过：看一个典型文件是否存在
        marker = None
        for n in zf.namelist():
            if n.lower().endswith(".jpg") or n.lower().endswith(".png"):
                marker = n
                break
        need_extract = True
        if marker is not None:
            test_path = os.path.join(EXTRACT_DIR, marker)
            if os.path.exists(test_path):
                need_extract = False
        if need_extract:
            zf.extractall(EXTRACT_DIR)
    # 返回本地根目录（一般会包含一个 mmdu_pics/ 子目录）
    # 解压后结构通常是 EXTRACT_DIR/mmdu_pics/xxx.jpg
    local_root = os.path.join(EXTRACT_DIR, "mmdu_pics")
    if not os.path.isdir(local_root):
        # 兜底：有的压缩包不带顶层目录，直接在根放图
        local_root = EXTRACT_DIR
    return os.path.abspath(local_root)

def map_to_local_paths(img_list, local_root):
    mapped = []
    for p in img_list:
        # 样本路径示例：/mmdu_pics/ABC.jpg
        # 去掉前导斜杠，拼到本地根
        sub = p.lstrip("/")  # mmdu_pics/ABC.jpg
        mapped.append(os.path.join(local_root, os.path.relpath(sub, "mmdu_pics")))
        # 上面 relpath 的目的：把 "mmdu_pics/..." 变成相对路径，防止重复拼接
    return mapped

def main():
    local_root = ensure_mmdu_pics_local()
    print("Local images root:", local_root)

    ds = load_dataset(REPO_ID, split="train")
    small = ds.select(range(5))

    # 新增一列：把原始的 '/mmdu_pics/xxx.jpg' 转成本地绝对路径
    def add_local(ex):
        return {"image_local": map_to_local_paths(ex["image"], local_root)}
    small = small.map(add_local, batched=False)

    # 把本地路径 cast 成图片序列
    small = small.cast_column("image_local", Sequence(Image()))

    print("Columns:", small.column_names)
    ex = small[0]
    print("num images in sample 0:", len(ex["image_local"]))
    img0 = ex["image_local"][0]
    print("first image size:", img0.size)

if __name__ == "__main__":
    main()
