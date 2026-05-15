# week4_ablations/clip_arrays_by_k.py
from pathlib import Path
import re, ast

VAR_DIR = Path("week4_ablations/variants_out")

# 只匹配一行的数组（你的文件目前是单行列表）；如果后续换成多行，改成 DOTALL 非贪婪版本即可
PAT = re.compile(r"^(SAMPLE\d+_(?:BLIP2|LLAVA|QWEN))\s*=\s*(\[[^\n]*\])", re.MULTILINE)

def clip_list_literal(lit: str, k: int) -> str:
    arr = ast.literal_eval(lit)  # 安全解析 0/1 列表
    if k <= 0:
        return "[]"
    clipped = arr[-k:]          # 如要“前 k 轮”，改成 arr[:k]
    return repr(clipped)

def main():
    for py in sorted(VAR_DIR.glob("recap_k*.py")):
        m = re.search(r"recap_k(\d+)\.py$", py.name)
        if not m:
            continue
        k = int(m.group(1))
        code = py.read_text()
        count = 0

        def repl(mm: re.Match) -> str:
            nonlocal count  # 这里在函数作用域内，合法
            name, lit = mm.group(1), mm.group(2)
            count += 1
            return f"{name} = {clip_list_literal(lit, k)}"

        new_code = PAT.sub(repl, code)
        out_py = py.with_name(py.stem + "_clipped.py")
        out_py.write_text(new_code)
        print(f"[OK] wrote {out_py.name}  k={k}, clipped {count} arrays")

    print("[DONE] all recap_k*.py clipped.")

if __name__ == "__main__":
    main()
