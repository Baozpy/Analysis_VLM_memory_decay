# -*- coding: utf-8 -*-
import importlib.util, types, re, json, os

MODEL_KEYS = ("BLIP2","LLAVA","QWEN")

def load_arrays(py_path):
    """"""
    spec = importlib.util.spec_from_file_location("samples_mod", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    pat = re.compile(r"^SAMPLE(\d+)_(TURN_TO_IMAGES|BLIP2|LLAVA|QWEN)$", re.I)
    tmp = {}
    for name in dir(mod):
        m = pat.match(name)
        if not m: 
            continue
        sid = int(m.group(1))
        key = m.group(2).upper()
        tmp.setdefault(sid, {"TURN_TO_IMAGES":None, "BLIP2":None, "LLAVA":None, "QWEN":None})
        tmp[sid][key] = getattr(mod, name)

    # 仅保留有 TURN_TO_IMAGES 的样本
    samples = {sid: pack for sid, pack in tmp.items() if pack["TURN_TO_IMAGES"] is not None}
    return samples

def write_arrays(py_path, samples, header="# Auto-generated week3 variant\n"):
    """把样本写成 SAMPLE{i}_* 变量的 .py 文件（供 stats_from_arrays/build_stats_from_samples.py --arrays 使用）"""
    os.makedirs(os.path.dirname(py_path), exist_ok=True)
    lines = [header, "\n"]
    for sid in sorted(samples.keys()):
        pack = samples[sid]
        def emit(name, val):
            lines.append(f"SAMPLE{sid}_{name} = {json.dumps(val, ensure_ascii=False)}\n")
        emit("TURN_TO_IMAGES", pack["TURN_TO_IMAGES"])
        for m in MODEL_KEYS:
            if pack[m] is not None:
                emit(m, pack[m])
        lines.append("\n")
    with open(py_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
