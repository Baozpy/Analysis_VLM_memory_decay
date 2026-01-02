# fit_half_life.py
import pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def decay_fn(x, p0, lam, c):
    # OK(Δ) = c + p0 * exp(-lam * Δ), clamp to [0,1]
    return np.clip(c + p0*np.exp(-lam*np.asarray(x)), 0, 1)

def fit_one(df_m):
    agg = df_m.groupby("delta")["ok"].mean().reset_index()
    x = agg["delta"].values
    y = agg["ok"].values
    # 初始值：p0≈y[0]-y[-1], lam≈0.2, c≈min(y)
    p0_0 = max(y[0]-y[-1], 0.05)
    p0 = [p0_0, 0.2, max(min(y[-3:]), 0.05)]
    popt,_ = curve_fit(decay_fn, x, y, p0=p0, bounds=([0,0,0],[1,5,1]))
    p0_hat, lam_hat, c_hat = popt
    # half-life
    t_half = np.log(2)/lam_hat if lam_hat>1e-8 else np.inf
    # 粗略R2
    yhat = decay_fn(x, *popt); r2 = 1-(((y-yhat)**2).sum()/((y-y.mean())**2).sum()+1e-9)
    return dict(p0=p0_hat, lam=lam_hat, c=c_hat, t_half=t_half, r2=r2, curve=agg)

if __name__=="__main__":
    df = pd.read_csv("stats_Δ.tsv", sep="\t")
    rows=[]
    for m, sub in df.groupby("model"):
        res = fit_one(sub)
        rows.append(dict(model=m, p0=res["p0"], lam=res["lam"], c=res["c"], half_life=res["t_half"], r2=res["r2"]))
        # plot
        x = res["curve"]["delta"].values
        y = res["curve"]["ok"].values
        xs = np.linspace(x.min(), x.max(), 200)
        yh = decay_fn(xs, res["p0"], res["lam"], res["c"])
        plt.figure()
        plt.scatter(x,y,label="empirical")
        plt.plot(xs,yh,label=f"fit: t½={res['t_half']:.2f}, R²={res['r2']:.2f}")
        plt.xlabel("Δ turns since first mention"); plt.ylabel("Answer-OK")
        plt.title(f"B2 decay fit - {m}"); plt.legend(); plt.tight_layout()
        plt.savefig(f"half_life_fit_{m}.png", dpi=160); plt.close()

    pd.DataFrame(rows).to_csv("half_life.tsv", sep="\t", index=False)
    print("Saved: half_life.tsv & figs.")
