#!/usr/bin/env python3
"""
Model 3: Richards growth (phenomenological)

  dC/dt = r·C·[1 - (C/K)^ν]

- adds ν to tune the curve’s sharpness at inflection.
- for ν≠1 it can mimic the CC downturn without a second ODE.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from pathlib import Path

# ---- settings ----
data_dir = Path("/Users/yousribouamara/Downloads/data varady")
data_files = {
    "231 alone":    "231alone_2.csv",
    "231 + M1 CM":  "231M1CM.csv",
    "231 + M2 CM":  "231M2CM.csv",
    "231 + M1 CC":  "231M1CC.csv",
    "231 + M2 CC":  "231M2CC.csv",
    "M1 alone CM":  "M1aloneCM.csv",
    "M2 alone CM":  "M2aloneCM.csv",
}
t_sim = np.linspace(0,72,200)
C0 = 1.0

# loader
def load_series(label):
    df = pd.read_csv(data_dir/data_files[label],
                     header=None, names=["Time","Fold"])
    return df["Time"].values, df["Fold"].values

# Richards ODE
def richards(C, t, r, K, ν):
    return r*C*(1 - (C/K)**ν)

def simulate(params, t):
    return odeint(richards, C0, t, args=tuple(params)).flatten()

# fit Richards for each condition
fit_params = {}
for lbl in data_files:
    t,y = load_series(lbl)
    def f(t, r, K, ν):
        return simulate((r,K,ν), t)

    p0 = [0.1,6,1.0]
    bnds = ([0,1,0.1],[1,100,5])
    p, _ = curve_fit(f, t, y, p0=p0, bounds=bnds)
    fit_params[lbl] = p
    print(f"[{lbl}] r={p[0]:.3f}, K={p[1]:.1f}, ν={p[2]:.3f}")

# plot groups
groups = {
    "CM alone":       ["M1 alone CM","M2 alone CM"],
    "Partial CM":     ["231 + M1 CM","231 + M2 CM"],
    "Co-culture CC":  ["231 + M1 CC","231 + M2 CC"],
}

for title, conds in groups.items():
    plt.figure(figsize=(7,5))
    plt.title(f"Model3—{title}")

    # baseline
    t0,y0 = load_series("231 alone")
    p0 = fit_params["231 alone"]
    plt.scatter(t0,y0,c='k',marker='o',label="231 alone data")
    plt.plot(t_sim, simulate(p0, t_sim),
             '--',c='k',label="231 alone model")

    # each condition
    for mk,col,lbl in zip(['s','^'],['b','r'],conds):
        t,y = load_series(lbl)
        plt.scatter(t,y,marker=mk,c=col,label=f"{lbl} data")
        p = fit_params[lbl]
        plt.plot(t_sim, simulate(p, t_sim),
                 '-',c=col,label=f"{lbl} model")

    plt.xlabel("Time (h)")
    plt.ylabel("Fold increase (ϕ)")
    plt.legend(bbox_to_anchor=(1,1),loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
