import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import importlib.util

# === Load model functions and parameters ===
spec = importlib.util.spec_from_file_location("MM_ODE", "MM_ODE.py")
MM_ODE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MM_ODE)

simulate_unified = MM_ODE.simulate_unified
r0_fit = MM_ODE.r0_fit  # DMEM, no need to downscale
Cmax_fit = MM_ODE.Cmax_fit

# === Load Pe data ===
df_ctrl = pd.read_csv("/Users/yousribouamara/Downloads/Data_Pe/control.csv")
df_tam  = pd.read_csv("/Users/yousribouamara/Downloads/Data_Pe/TAM.csv")
df_m2   = pd.read_csv("/Users/yousribouamara/Downloads/Data_Pe/M2.csv")

# === Normalize to t=0 for fold increase ===
def normalize(df):
    df = df.copy()
    base = df["FoldIncrease"].iloc[0]
    df["FoldIncrease"] = df["FoldIncrease"] / base
    df["Error"] = df["Error"] / base
    return df

df_ctrl = normalize(df_ctrl)
df_tam  = normalize(df_tam)
df_m2   = normalize(df_m2)

# === Simulation setup ===
timepoints = df_ctrl["Time"].values
C0 = 1.0

params = {
    "K1": 1.0,
    "K2": 1.0,
    "E1": 0.6,
    "E2": 0.4,
    "alpha": 0.0
}

# Simulate each condition
ctrl_sim = simulate_unified(timepoints, r0_fit, 0.0, 0.0, **params, Cmax=Cmax_fit, C0=C0)
m2_sim   = simulate_unified(timepoints, r0_fit, 0.0, 1.0, **params, Cmax=Cmax_fit, C0=C0)
tam_sim  = simulate_unified(timepoints, r0_fit, 0.50, 0.50, **params, Cmax=Cmax_fit, C0=C0)

# === Plotting ===
plt.figure(figsize=(10, 6))

# --- Experimental data: scatter with error bars (no lines) ---
plt.errorbar(df_ctrl["Time"], df_ctrl["FoldIncrease"], yerr=df_ctrl["Error"],
             fmt='o', color='gray', label='Pe Control (exp)', capsize=4)
plt.errorbar(df_m2["Time"], df_m2["FoldIncrease"], yerr=df_m2["Error"],
             fmt='s', color='red', label='Pe M2 (exp)', capsize=4)
plt.errorbar(df_tam["Time"], df_tam["FoldIncrease"], yerr=df_tam["Error"],
             fmt='^', color='blue', label='Pe TAM (exp)', capsize=4)

# --- Model predictions: dashed lines only ---
plt.plot(timepoints, ctrl_sim, '--', color='gray', label='Model: Control')
plt.plot(timepoints, m2_sim, '--', color='red', label='Model: M2')
plt.plot(timepoints, tam_sim, '--', color='blue', label='Model: TAM (55% M1)')

# Labels and layout
plt.title("Validation of Unified Model on Pe et al. (2022) Data")
plt.xlabel("Time (hours)")
plt.ylabel("Fold Increase (normalized to t=0)")
plt.xticks([24, 48, 72])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
