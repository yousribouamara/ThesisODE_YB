import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---- settings ----
data_dir = Path("/Users/yousribouamara/Downloads/data varady")
data_files = {
    "231 alone 2X": "231alone2X.csv",
    "231 alone": "231alone_2.csv",
    "231 + M1 CC": "231M1CC.csv",
    "231 + M2 CC": "231M2CC.csv",
    "231 + M1 CM": "231M1CM.csv",
    "231 + M2 CM": "231M2CM.csv",
    "M1 alone CM": "M1aloneCM.csv",
    "M2 alone CM": "M2aloneCM.csv",
}

# which two baselines to always include
baseline = ["231 alone 2X", "231 alone"]

# define the three groups you want to plot
groups = {
    "Co-culture (CC)": ["231 + M1 CC", "231 + M2 CC"],
    "Partial co-culture (CM)": ["231 + M1 CM", "231 + M2 CM"],
    "Macrophage alone (CM)": ["M1 alone CM", "M2 alone CM"],
}


# helper to load one series
def load_series(label):
    df = pd.read_csv(
        data_dir / data_files[label],
        header=None,
        names=["Time", "Fold"]
    )
    return df["Time"], df["Fold"]


# per-figure marker/color palette (exactly 4 entries)
fig_markers = ["o", "s", "^", "D"]
fig_colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]

# ---- make each of the 3 figures ----
for title, conds in groups.items():
    plt.figure(figsize=(7, 5))
    plt.title(title)

    # build the list of labels we'll plot in this figure
    labels_to_plot = baseline + conds

    # for each one, assign its marker & color by zipping into fig_markers/colors
    for lbl, mk, col in zip(labels_to_plot, fig_markers, fig_colors):
        t, y = load_series(lbl)
        plt.plot(
            t, y,
            label=lbl,
            marker=mk,
            color=col,
            linestyle='-',
            linewidth=2,
            markersize=6
        )

    plt.xlabel("Time (hours)")
    plt.ylabel("Normalized fold increase (ϕ)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
