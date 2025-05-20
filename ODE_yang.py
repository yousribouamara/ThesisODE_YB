# validate_yang_data.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import importlib.util

# --- Load the existing MM_ODE.py (Varady-calibrated model) ---
spec = importlib.util.spec_from_file_location("MM_ODE", "MM_ODE.py")
MM_ODE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MM_ODE)

# --- Load Yang et al. data from figure 2D ---
df_raw = pd.read_csv("/Users/yousribouamara/Downloads/wpd_datasets.csv", skiprows=1)
yang_raw = pd.DataFrame({
    "Time": df_raw["X.1"].astype(float),
    "M1": df_raw["Y.1"].astype(float),
    "M2": df_raw["Y.2"].astype(float),
    "Ctrl": df_raw["Y.4"].astype(float)
}).dropna()

# --- Normalize to time 0 (fold increase relative to first timepoint) ---
yang_norm = yang_raw.copy()
for col in ['M1', 'M2', 'Ctrl']:
    yang_norm[col] = yang_norm[col] / yang_norm[col].iloc[0]

# --- Use your Varady model with minor correction for HMM growth medium ---
r0_adj = MM_ODE.r0_fit * 0.6  # slower proliferation in HMM, this is just a guess
Cmax = MM_ODE.Cmax_fit
C0 = 1.0
t = yang_norm["Time"].values

# Use original E1/E2 estimates — don't fit anything here
params = {
    'K1': 1.0,
    'K2': 1.0,
    'E1': 0.6,
    'E2': 0.4
}

# --- Define model (copied structure to avoid replotting MM_ODE's internal calls) ---
def unified_model(C, t, r0, M1, M2, K1, K2, E1, E2, Cmax, alpha=0.0):
    inhib_M1 = E1 * M1 / (K1 + M1)
    inhib_M2 = E2 * M2 / (K2 + M2)
    combined = 1 - (1 - inhib_M1) * (1 - inhib_M2) + alpha * inhib_M1 * inhib_M2
    combined = np.clip(combined, 0, 1)
    r_eff = r0 * (1 - combined)
    return r_eff * C * (1 - C / Cmax)

def simulate_unified(t, r0, M1, M2, K1, K2, E1, E2, Cmax, alpha=0.0, C0=1.0):
    sol = odeint(unified_model, C0, t, args=(r0, M1, M2, K1, K2, E1, E2, Cmax, alpha))
    return sol.ravel()

# --- Simulate model for M1, M2, and Ctrl conditions ---
sim_ctrl = simulate_unified(t, r0_adj, 0, 0, **params, Cmax=Cmax, C0=C0)
sim_M1   = simulate_unified(t, r0_adj, 1, 0, **params, Cmax=Cmax, C0=C0)
sim_M2   = simulate_unified(t, r0_adj, 0, 1, **params, Cmax=Cmax, C0=C0)

# --- Plot 1: Yang normalized data only ---
plt.figure(figsize=(8, 5))
plt.plot(t, yang_norm["Ctrl"], label="231 (Ctrl)", color='gray', marker='o')
plt.plot(t, yang_norm["M1"], label="231 + M1", color='blue', marker='s')
plt.plot(t, yang_norm["M2"], label="231 + M2", color='red', marker='^')
plt.title("Yang et al. (2016): Normalized Fold Increase (HMM)")
plt.xlabel("Time (h)")
plt.ylabel("Fold Increase (normalized)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 2: Model vs Yang data comparison ---
plt.figure(figsize=(10, 6))
plt.plot(t, sim_ctrl, '--', label='Model: 231 alone', color='gray')
plt.plot(t, sim_M1, label='Model: +M1', color='blue')
plt.plot(t, sim_M2, label='Model: +M2', color='red')

plt.scatter(t, yang_norm["Ctrl"], label='Yang: 231 alone', color='gray', marker='o')
plt.scatter(t, yang_norm["M1"], label='Yang: +M1', color='blue', marker='s')
plt.scatter(t, yang_norm["M2"], label='Yang: +M2', color='red', marker='^')

plt.title("Model vs. Yang et al. (2016): Fold Increase Validation")
plt.xlabel("Time (h)")
plt.ylabel("Normalized Fold Increase")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
