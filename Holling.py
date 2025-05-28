#!/usr/bin/env python3
"""
Model 1: Logistic growth + saturating Holling-II kill.

  dC/dt = r·C·(1 - C/K)             # logistic self-limitation
          - α·M·C/(h + C)           # macrophage kill that saturates

This kill term bends the CC curves downward at high confluence,
matching your data’s shape much better than plain exponential.
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

# assume “dose” M=1 for any macrophage condition, M=0 for baseline
M_vals = {lbl: (0.0 if lbl=="231 alone" else 1.0)
          for lbl in data_files}

t_sim = np.linspace(0, 72, 200)
C0 = 1.0

# load helper
def load_series(label):
    df = pd.read_csv(data_dir/data_files[label],
                     header=None, names=["Time","Fold"])
    return df["Time"].values, df["Fold"].values

# model + simulator
def holling(C, t, r, K, α, h, M):
    growth = r*C*(1 - C/K)
    kill   = α*M*C/(h + C)
    return growth - kill

def simulate(params, t):
    return odeint(holling, C0, t, args=tuple(params)).flatten()

# 1) fit baseline logistic (alpha/h = 0)
t0, y0 = load_series("231 alone")
def pure_logistic(t, r, K):
    return odeint(lambda C,t: r*C*(1 - C/K), C0, t).flatten()

(r0, K0), _ = curve_fit(pure_logistic, t0, y0,
                        p0=[0.1,6.0], bounds=([0,1],[1,100]))
print(f"[baseline] r={r0:.3f}, K={K0:.1f}")

# 2) fit α,h for every other condition
fit_params = {}
for lbl in data_files:
    if lbl == "231 alone":
        continue
    t, y = load_series(lbl)
    M = M_vals[lbl]

    def fit_fun(t, α, h):
        return simulate((r0, K0, α, h, M), t)

    p0 = [0.2,1.0]
    bnds = ([0,0.1],[10,10])
    (α_fit, h_fit), _ = curve_fit(fit_fun, t, y, p0=p0, bounds=bnds)
    fit_params[lbl] = (r0, K0, α_fit, h_fit, M)
    print(f"[{lbl}] α={α_fit:.3f}, h={h_fit:.3f}")

# 3) group-by-group plotting
groups = {
    "CM alone":       ["M1 alone CM", "M2 alone CM"],
    "Partial CM on 231": ["231 + M1 CM", "231 + M2 CM"],
    "Co-culture (CC)":    ["231 + M1 CC", "231 + M2 CC"],
}

for title, conds in groups.items():
    plt.figure(figsize=(7,5))
    plt.title(f"Model1—{title}")

    # baseline data + model
    t_base, y_base = load_series("231 alone")
    plt.scatter(t_base, y_base, c="k", marker="o", label="231 alone data")
    plt.plot(t_sim, pure_logistic(t_sim, r0, K0),
             "--", c="k", label="231 alone model")

    # each condition
    for mk, col, lbl in zip(["s","^"], ["tab:blue","tab:red"], conds):
        t, y = load_series(lbl)
        plt.scatter(t, y, marker=mk, color=col, label=f"{lbl} data")
        params = fit_params[lbl]
        y_sim = simulate(params, t_sim)
        plt.plot(t_sim, y_sim, "-", color=col, label=f"{lbl} model")

    plt.xlabel("Time (h)")
    plt.ylabel("Fold increase (ϕ)")
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
