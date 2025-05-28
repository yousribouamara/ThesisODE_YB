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

dataframes = {
    label: pd.read_csv(os.path.join(path, filename), header=None, names=["Time", "FoldIncrease"])
    for label, filename in data_files.items()
}

# --- Unified ODE Model with alpha (antagonism control) ---
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

# --- STEP 1: Fit baseline
df_alone = dataframes["231 alone"]
t_data = df_alone["Time"].values
y_data = df_alone["FoldIncrease"].values

def fit_231_alone(t, r0, Cmax):
    return simulate_unified(t, r0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, Cmax)

params_alone, _ = curve_fit(fit_231_alone, t_data, y_data, p0=[0.05, 5.0], bounds=([0, 1], [1, 100]))
r0_fit, Cmax_fit = params_alone

# --- STEP 2: Fit macrophage effects
fit_results = {}
condition_mappings = {
    "231 + M1 CM":   (1.0, 0.0),
    "231 + M2 CM":   (0.0, 1.0),
    "231 + M1 CC":   (1.0, 0.0),
    "231 + M2 CC":   (0.0, 1.0),
    "M1 alone CM":   (1.0, 0.0),
    "M2 alone CM":   (0.0, 1.0),
}

def make_fit_function(M1_fixed, M2_fixed):
    def fit_func(t, K1, K2, E1, E2):
        return simulate_unified(t, r0_fit, M1_fixed, M2_fixed, K1, K2, E1, E2, Cmax_fit)
    return fit_func

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

# --- STEP 3: Plot data and model fits
t_sim = np.linspace(0, 72, 100)
C0 = 1.0

plt.figure(figsize=(12, 8))
for label in dataframes:
    df = dataframes[label]
    t_data = df["Time"].values
    y_data = df["FoldIncrease"].values
    plt.scatter(t_data, y_data, label=f"{label} (data)")
    if label == "231 alone":
        C_sim = simulate_unified(t_sim, r0_fit, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, Cmax_fit)
    elif label in fit_results and fit_results[label] != "Fit failed":
        res = fit_results[label]
        C_sim = simulate_unified(t_sim, r0_fit, res["M1"], res["M2"],
                                 res["K1"], res["K2"], res["E1"], res["E2"], Cmax_fit)
    else:
        continue
    plt.plot(t_sim, C_sim, linestyle='--', label=f"{label} (model)")

plt.xlabel("Time (hours)")
plt.ylabel("Normalized Fold Increase")
plt.title("Cancer Cell Growth with Macrophage Dose Effects")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- STEP 4: Mixed condition with antagonism
M1_custom = 0.5
M2_custom = 0.5
alpha = -0.5  #Tuned antagonism factor

custom_contexts = {
    "CM only (Mφ alone)":       ("M1 alone CM", "M2 alone CM"),
    "Partial CM (231 + Mφ CM)": ("231 + M1 CM", "231 + M2 CM"),
    "Full Co-culture":          ("231 + M1 CC", "231 + M2 CC"),
}

df_control = dataframes["231 alone"]
t_control = df_control["Time"].values
y_control = df_control["FoldIncrease"].values
C_control_sim = simulate_unified(t_sim, r0_fit, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, Cmax_fit)

for context_name, (m1_key, m2_key) in custom_contexts.items():
    if fit_results.get(m1_key) == "Fit failed" or fit_results.get(m2_key) == "Fit failed":
        continue
    m1 = fit_results[m1_key]
    m2 = fit_results[m2_key]

    C_M1_sim = simulate_unified(t_sim, r0_fit, 1.0, 0.0, m1["K1"], m1["K2"], m1["E1"], m1["E2"], Cmax_fit)
    C_M2_sim = simulate_unified(t_sim, r0_fit, 0.0, 1.0, m2["K1"], m2["K2"], m2["E1"], m2["E2"], Cmax_fit)

    C_mix_sim = simulate_unified(
        t_sim,
        r0_fit,
        M1_custom,
        M2_custom,
        m1["K1"],
        m2["K2"],
        m1["E1"],
        m2["E2"],
        Cmax_fit,
        alpha=alpha
    )

    plt.figure(figsize=(10, 6))
    plt.scatter(t_control, y_control, label="231 alone (data)", color="gray", zorder=5)
    plt.plot(t_sim, C_control_sim, label="231 alone (model)", color="gray", linestyle="dotted")
    plt.plot(t_sim, C_M1_sim, label="100% M1", color="blue")
    plt.plot(t_sim, C_M2_sim, label="100% M2", color="red")
    plt.plot(t_sim, C_mix_sim, label=f"Custom mix 50/50 (α={alpha})", color="green", linestyle="--")

    plt.title(f"Simulated Growth – {context_name}")
    plt.xlabel("Time (hours)")
    plt.ylabel("Normalized Fold Increase")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- STEP 5: Michaelis-Menten inhibition curves
doses = np.linspace(0, 2, 200)
M1_K = fit_results["M1 alone CM"]["K1"]
M1_E = fit_results["M1 alone CM"]["E1"]
M2_K = fit_results["M2 alone CM"]["K2"]
M2_E = fit_results["M2 alone CM"]["E2"]

inhib_M1 = M1_E * doses / (M1_K + doses)
inhib_M2 = M2_E * doses / (M2_K + doses)

plt.figure(figsize=(8, 6))
plt.plot(doses, inhib_M1, label=f"M1 inhibition\n(E={M1_E:.2f}, K={M1_K:.2f})", color='blue')
plt.plot(doses, inhib_M2, label=f"M2 inhibition\n(E={M2_E:.2f}, K={M2_K:.2f})", color='red')
plt.xlabel("Macrophage dose (arbitrary units)")
plt.ylabel("Inhibition contribution")
plt.title("Michaelis-Menten Inhibition Curves for M1 and M2 Macrophages")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
