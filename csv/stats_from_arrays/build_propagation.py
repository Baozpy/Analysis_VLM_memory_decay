# build_propagation.py
import pandas as pd
from ast import literal_eval
from pathlib import Path

def explode_events(df):
    # df: sample_id, turn, images(str or list), ok, model
    if df["images"].dtype==object and isinstance(df["images"].iloc[0], str):
        df = df.copy(); df["images"] = df["images"].apply(lambda x: literal_eval(x) if isinstance(x,str) else x)
    rows=[]
    for r in df.itertuples():
        for i in r.images:
            rows.append((r.sample_id, r.turn, i, r.ok, r.model))
    return pd.DataFrame(rows, columns=["sample_id","turn","image_id","ok","model"])

def propagation_one(df_m):
    # 对每个 (s, i) 找首错 t*，向后看连续受影响长度 L
    out=[]
    for (s,i), g in df_m.sort_values("turn").groupby(["sample_id","image_id"]):
        g = g.sort_values("turn")
        if (g.ok==0).any():
            t_star = g[g.ok==0].turn.iloc[0]
            after = g[g.turn>t_star]
            L = (after.ok==0).sum()    # 简单定义：后续错误数
            first_k = after.turn.iloc[0]-t_star if len(after)>0 else 0
            out.append(dict(sample_id=s, image_id=i, t_star=t_star, tail_errors=L, first_gap=first_k))
    return pd.DataFrame(out)

if __name__=="__main__":
    # 期望输入 tsv: sample_id, turn, images(如"[1,2]"), ok(0/1), model
    df = pd.read_csv("events.tsv", sep="\t")
    ev = explode_events(df)
    rows=[]
    edges=[]
    for m, sub in ev.groupby("model"):
        prop = propagation_one(sub)
        prop["model"] = m
        rows.append(prop)
        # 边：从 t* 指向后续错误 turn（可选，如需画图）
        for (s,i), g in sub.groupby(["sample_id","image_id"]):
            g = g.sort_values("turn")
            bad = g[g.ok==0]
            if bad.empty: continue
            t_star = bad.turn.iloc[0]
            for t in bad.turn[bad.turn>t_star]:
                edges.append((m, s, int(i), int(t_star), int(t)))
    pd.concat(rows).to_csv("propagation_stats.tsv", sep="\t", index=False)
    pd.DataFrame(edges, columns=["model","sample_id","image_id","t_star","t_bad"]).to_csv("prop_edges.tsv", sep="\t", index=False)
    print("Saved: propagation_stats.tsv, prop_edges.tsv")
