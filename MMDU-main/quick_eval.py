# quick_eval.py
import csv, re, sys
from collections import Counter

def ngrams(seq, n):
    return [' '.join(seq[i:i+n]) for i in range(len(seq)-n+1)]

def echo_rate(prompt, answer):
    p = re.findall(r"\w+", prompt.lower())
    a = re.findall(r"\w+", answer.lower())
    if not a: return 0.0
    pset = set(p)
    overlap = sum(1 for w in a if w in pset)
    return overlap/len(a)

def rep_rate(answer, n=3):
    a = re.findall(r"\w+", answer.lower())
    gs = ngrams(a, n)
    if not gs: return 0.0
    cnt = Counter(gs)
    rep = sum(c for g,c in cnt.items() if c>1)
    return rep/len(gs)

in_csv = sys.argv[1] if len(sys.argv)>1 else "pilot_results.csv"
out_csv = "pilot_eval.csv"

with open(in_csv, newline="", encoding="utf-8") as f, open(out_csv, "w", newline="", encoding="utf-8") as g:
    r = csv.DictReader(f)
    w = csv.writer(g)
    w.writerow(["sample_id","turn","img_idx","EchoRate","Repetition3g","AnsLen"])
    for row in r:
        e = echo_rate(row["prompt"], row["answer"])
        rep = rep_rate(row["answer"], 3)
        alen = len(re.findall(r"\w+", row["answer"]))
        w.writerow([row["sample_id"], row["turn"], row["img_idx"], f"{e:.3f}", f"{rep:.3f}", alen])

print("✅ Saved to", out_csv)
