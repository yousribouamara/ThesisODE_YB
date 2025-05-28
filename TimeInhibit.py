#!/usr/bin/env python3
"""
Model 2: Two-ODE inhibitor buildup.

  dI/dt = s·M – d·I
  dC/dt = r·C·(1 – C/K) – β·I·C

I(t) starts at zero and accumulates, creating a delayed-but-strong inhibition
that bends your CC curves downward naturally.
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
M_vals = {lbl:(0.0 if lbl=="231 alone" else 1.0)
          for lbl in data_files}

t_sim = np.linspace(0,72,200)
C0, I0 = 1.0, 0.0

# loader
def load_series(label):
    df = pd.read_csv(data_dir/data_files[label],
                     header=None, names=["Time","Fold"])
    return df["Time"].values, df["Fold"].values

# system of ODEs
def system(X, t, r, K, s, d, β, M):
    I,C = X
    dI = s*M - d*I
    dC = r*C*(1 - C/K) - β*I*C
    return [dI, dC]

def simulate(params, t):
    sol = odeint(system, [I0,C0], t, args=tuple(params))
    return sol[:,1]  # return C only

# 1) fit baseline logistic
t0,y0 = load_series("231 alone")
def base_log(t, r, K):
    return odeint(lambda C,t: r*C*(1-C/K), C0, t).flatten()

(r0, K0), _ = curve_fit(base_log, t0, y0,
                        p0=[0.1,6], bounds=([0,1],[1,100]))
print(f"[baseline] r={r0:.3f}, K={K0:.1f}")

# 2) fit s,d,β for each condition
fit_params = {}
for lbl in data_files:
    if lbl == "231 alone":
        continue
    t,y = load_series(lbl); M = M_vals[lbl]

    def fit_fun(t, s, d, β):
        return simulate((r0,K0,s,d,β,M), t)

    p0 = [0.5,0.1,0.2]
    bnds = ([0,0,0],[10,1,1])
    (s_f,d_f,β_f), _ = curve_fit(fit_fun, t,y, p0=p0, bounds=bnds)
    fit_params[lbl] = (r0,K0,s_f,d_f,β_f,M)
    print(f"[{lbl}] s={s_f:.3f}, d={d_f:.3f}, β={β_f:.3f}")

# 3) plot groups
groups = {
    "CM alone":       ["M1 alone CM","M2 alone CM"],
    "Partial CM":     ["231 + M1 CM","231 + M2 CM"],
    "Co-culture CC":  ["231 + M1 CC","231 + M2 CC"],
}

for title, conds in groups.items():
    plt.figure(figsize=(7,5))
    plt.title(f"Model2—{title}")

    # baseline
    t0,y0 = load_series("231 alone")
    plt.scatter(t0,y0,c='k',marker='o',label="231 alone data")
    plt.plot(t_sim, base_log(t_sim,r0,K0),
             '--',c='k',label="231 alone model")

    # each condition
    for mk,col,lbl in zip(['s','^'],['b','r'],conds):
        t,y = load_series(lbl)
        plt.scatter(t,y,marker=mk,c=col,label=f"{lbl} data")
        params = fit_params[lbl]
        plt.plot(t_sim, simulate(params, t_sim),
                 '-',c=col,label=f"{lbl} model")

    plt.xlabel("Time (h)")
    plt.ylabel("Fold increase (ϕ)")
    plt.legend(bbox_to_anchor=(1,1),loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
