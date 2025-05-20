import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import os

# --- Load Data ---

path = "/Users/yousribouamara/Downloads/data varady"
data_files = {
    "231 alone":     "231alone_2.csv",
    "231 + M1 CM":   "231M1CM.csv",
    "231 + M1 CC":   "231M1CC.csv",
    "231 + M2 CM":   "231M2CM.csv",
    "231 + M2 CC":   "231M2CC.csv",
    "M1 alone CM":   "M1aloneCM.csv",
    "M2 alone CM":   "M2aloneCM.csv",
}

# Read all files into a dictionary
dataframes = {
    label: pd.read_csv(os.path.join(path, filename), header=None, names=["Time", "FoldIncrease"])
    for label, filename in data_files.items()
}

# --- Define the Unified ODE Model ---
def unified_model(C, t, r0, M1, M2, K1, K2, E1, E2, Cmax):
    # Saturating inhibition by M1 and M2 macrophages
    inhib_M1 = E1 * M1 / (K1 + M1)
    inhib_M2 = E2 * M2 / (K2 + M2)
    r_eff = r0 * (1 - inhib_M1 - inhib_M2)  # Effective growth rate
    return r_eff * C * (1 - C / Cmax)       # Logistic growth term

# Wrapper to integrate the model
def simulate_unified(t, r0, M1, M2, K1, K2, E1, E2, Cmax, C0=1.0):
    sol = odeint(unified_model, C0, t, args=(r0, M1, M2, K1, K2, E1, E2, Cmax))
    return sol.ravel()

# --- STEP 1: Fit the baseline growth: "231 alone" ---
df_alone = dataframes["231 alone"]
t_data = df_alone["Time"].values
y_data = df_alone["FoldIncrease"].values

# This function fits only r0 and Cmax since M1 = M2 = 0 in this condition
def fit_231_alone(t, r0, Cmax):
    return simulate_unified(t, r0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, Cmax)

# Curve fitting on baseline data
params_alone, _ = curve_fit(fit_231_alone, t_data, y_data, p0=[0.05, 5.0], bounds=([0, 1], [1, 100]))
r0_fit, Cmax_fit = params_alone  # Save baseline rate and max capacity

# --- STEP 2: Fit macrophage conditions ---
fit_results = {}  # Dictionary to store fit parameters

# Define experimental macrophage doses (arbitrary units)
condition_mappings = {
    "231 + M1 CM":   (1.0, 0.0),
    "231 + M2 CM":   (0.0, 1.0),
    "231 + M1 CC":   (1.0, 0.0),
    "231 + M2 CC":   (0.0, 1.0),
    "M1 alone CM":   (1.0, 0.0),
    "M2 alone CM":   (0.0, 1.0),
}

# Closure to generate fitting function for specific (M1, M2)
def make_fit_function(M1_fixed, M2_fixed):
    def fit_func(t, K1, K2, E1, E2):
        return simulate_unified(t, r0_fit, M1_fixed, M2_fixed, K1, K2, E1, E2, Cmax_fit)
    return fit_func

# Loop through each condition and fit its parameters
for label, (M1_fixed, M2_fixed) in condition_mappings.items():
    df = dataframes[label]
    t_data = df["Time"].values
    y_data = df["FoldIncrease"].values
    try:
        fit_func = make_fit_function(M1_fixed, M2_fixed)
        params, _ = curve_fit(fit_func, t_data, y_data, p0=[1.0, 1.0, 0.5, 0.5],
                              bounds=([0.01, 0.01, 0, 0], [10, 10, 1, 1]))
        fit_results[label] = {
            "M1": M1_fixed,
            "M2": M2_fixed,
            "K1": params[0],
            "K2": params[1],
            "E1": params[2],
            "E2": params[3]
        }
    except RuntimeError:
        fit_results[label] = "Fit failed"

# --- STEP 3: Plot all results ---
t_sim = np.linspace(0, 72, 100)
C0 = 1.0

plt.figure(figsize=(12, 8))

# Loop through all datasets and simulate or fit
for label in dataframes:
    df = dataframes[label]
    t_data = df["Time"].values
    y_data = df["FoldIncrease"].values

    # Plot experimental data points
    plt.scatter(t_data, y_data, label=f"{label} (data)")

    # Simulate based on condition
    if label == "231 alone":
        # No M1 or M2
        C_sim = simulate_unified(t_sim, r0_fit, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, Cmax_fit)
    elif label in fit_results and fit_results[label] != "Fit failed":
        res = fit_results[label]
        C_sim = simulate_unified(t_sim, r0_fit, res["M1"], res["M2"],
                                 res["K1"], res["K2"], res["E1"], res["E2"], Cmax_fit)
    else:
        continue

    # Plot simulation result
    plt.plot(t_sim, C_sim, linestyle='--', label=f"{label} (model)")

# Finalize plot
plt.xlabel("Time (hours)")
plt.ylabel("Normalized Fold Increase")
plt.title("Cancer Cell Growth with Macrophage Dose Effects")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --- STEP 4: Simulate a custom hypothetical situation and compare with M1-only, M2-only, and 231 alone ---

# Define a new M1/M2 ratio: 20% M1, 80% M2
M1_custom = 0.5
M2_custom = 0.5

# Define representative contexts
custom_contexts = {
    "CM only (Mφ alone)":       ("M1 alone CM", "M2 alone CM"),
    "Partial CM (231 + Mφ CM)": ("231 + M1 CM", "231 + M2 CM"),
    "Full Co-culture":          ("231 + M1 CC", "231 + M2 CC"),
}

# Pull 231 alone data once
df_control = dataframes["231 alone"]
t_control = df_control["Time"].values
y_control = df_control["FoldIncrease"].values
C_control_sim = simulate_unified(t_sim, r0_fit, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, Cmax_fit)

# Loop through the three contexts
for context_name, (m1_key, m2_key) in custom_contexts.items():
    if fit_results.get(m1_key) == "Fit failed" or fit_results.get(m2_key) == "Fit failed":
        continue

    # Get parameter sets for 100% M1 and 100% M2
    m1 = fit_results[m1_key]
    m2 = fit_results[m2_key]

    # --- Simulate all three conditions ---
    C_M1_sim = simulate_unified(t_sim, r0_fit, 1.0, 0.0, m1["K1"], m1["K2"], m1["E1"], m1["E2"], Cmax_fit)
    C_M2_sim = simulate_unified(t_sim, r0_fit, 0.0, 1.0, m2["K1"], m2["K2"], m2["E1"], m2["E2"], Cmax_fit)

    # Weighted parameters for custom mixed case
    K1_mix = M1_custom * m1["K1"] + M2_custom * m2["K1"]
    K2_mix = M1_custom * m1["K2"] + M2_custom * m2["K2"]
    E1_mix = M1_custom * m1["E1"] + M2_custom * m2["E1"]
    E2_mix = M1_custom * m1["E2"] + M2_custom * m2["E2"]

    C_mix_sim = simulate_unified(t_sim, r0_fit, M1_custom, M2_custom, K1_mix, K2_mix, E1_mix, E2_mix, Cmax_fit)

    # --- Plot figure for this category ---
    plt.figure(figsize=(10, 6))
    plt.scatter(t_control, y_control, label="231 alone (data)", color="gray", zorder=5)
    plt.plot(t_sim, C_control_sim, label="231 alone (model)", color="gray", linestyle="dotted")

    plt.plot(t_sim, C_M1_sim, label="100% M1", color="blue")
    plt.plot(t_sim, C_M2_sim, label="100% M2", color="red")
    plt.plot(t_sim, C_mix_sim, label=f"Custom mix: {int(M1_custom*100)}% M1 / {int(M2_custom*100)}% M2", color="green", linestyle="--")

    plt.title(f"Simulated Growth – {context_name}")
    plt.xlabel("Time (hours)")
    plt.ylabel("Normalized Fold Increase")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
