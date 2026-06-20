import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ===========================
# 1. Load data manually
# ===========================

N_TIME = 600
N_ENSEMBLE = 10000

x_truth = pd.read_csv(
    "x_truth_trajectory_seed30.csv"
).iloc[:N_TIME, 1:].to_numpy()  # shape: (500, 40)


# EnSF-LR outputs
ensf_lr_x0_forecast = pd.read_csv(
    "x_0_traj_forecast_hristo_EnSF_nonlinear_LR_25%obs_10obsgap_seed30.csv"
).iloc[:N_TIME, 1:].to_numpy()

ensf_lr_x0_analysis = pd.read_csv(
    "x_0_traj_hristo_EnSF_nonlinear_LR_25%obs_10obsgap_seed30.csv"
).iloc[:N_TIME, 1:].to_numpy()

ensf_lr_x1_forecast = pd.read_csv(
    "x_1_traj_forecast_hristo_EnSF_nonlinear_LR_25%obs_10obsgap_seed30.csv"
).iloc[:N_TIME, 1:].to_numpy()

ensf_lr_x1_analysis = pd.read_csv(
    "x_1_traj_hristo_EnSF_nonlinear_LR_25%obs_10obsgap_seed30.csv"
).iloc[:N_TIME, 1:].to_numpy()


# EnKF outputs
enkf_x0_forecast = pd.read_csv(
    "x_0_traj_forecast_hristo_EnKF_nonlinear_25%obs_10obsgap_seed30.csv"
).iloc[:N_TIME, 1:].to_numpy()

enkf_x0_analysis = pd.read_csv(
    "x_0_traj_hristo_EnKF_nonlinear_25%obs_10obsgap_seed30.csv"
).iloc[:N_TIME, 1:].to_numpy()

enkf_x1_forecast = pd.read_csv(
    "x_1_traj_forecast_hristo_EnKF_nonlinear_25%obs_10obsgap_seed30.csv"
).iloc[:N_TIME, 1:].to_numpy()

enkf_x1_analysis = pd.read_csv(
    "x_1_traj_hristo_EnKF_nonlinear_25%obs_10obsgap_seed30.csv"
).iloc[:N_TIME, 1:].to_numpy()


# ===========================
# 2. Configuration
# ===========================

time_steps = [100, 200, 300, 400, 500]
subsample_size = 2000
fontsize = 16

rng = np.random.default_rng(42)
subsample_idx = rng.choice(N_ENSEMBLE, size=subsample_size, replace=False)

colors = {
    "forecast": "#9ecae1",    # light blue
    "analysis": "#d62728",    # red
    "truth": "#ffd700",       # gold/yellow
}


# ===========================
# 3. Compute shared axis limits
# ===========================

all_x0 = []
all_x1 = []

for t in time_steps:
    # EnSF-LR
    all_x0.extend(ensf_lr_x0_forecast[t, subsample_idx])
    all_x0.extend(ensf_lr_x0_analysis[t, subsample_idx])
    all_x1.extend(ensf_lr_x1_forecast[t, subsample_idx])
    all_x1.extend(ensf_lr_x1_analysis[t, subsample_idx])

    # EnKF
    all_x0.extend(enkf_x0_forecast[t, subsample_idx])
    all_x0.extend(enkf_x0_analysis[t, subsample_idx])
    all_x1.extend(enkf_x1_forecast[t, subsample_idx])
    all_x1.extend(enkf_x1_analysis[t, subsample_idx])

    # Truth
    all_x0.append(x_truth[t, 0])
    all_x1.append(x_truth[t, 1])

all_x0 = np.asarray(all_x0)
all_x1 = np.asarray(all_x1)

margin = 0.12

x_min, x_max = np.nanmin(all_x0), np.nanmax(all_x0)
y_min, y_max = np.nanmin(all_x1), np.nanmax(all_x1)

x_range = x_max - x_min
y_range = y_max - y_min

xlim = (x_min - margin * x_range, x_max + margin * x_range)
ylim = (y_min - margin * y_range, y_max + margin * y_range)


# ===========================
# 4. Plot helper
# ===========================

def plot_single_panel(
    ax,
    forecast_x0,
    forecast_x1,
    analysis_x0,
    analysis_x1,
    truth_x0,
    truth_x1,
    time_step,
    panel_label,
    show_xlabel=False,
    show_ylabel=False,
):
    """Plot forecast ensemble, analysis ensemble, and truth in the x0-x1 plane."""

    ax.scatter(
        forecast_x0[subsample_idx],
        forecast_x1[subsample_idx],
        color=colors["forecast"],
        s=14,
        alpha=0.25,
        edgecolors="none",
        zorder=1,
    )

    ax.scatter(
        analysis_x0[subsample_idx],
        analysis_x1[subsample_idx],
        color=colors["analysis"],
        s=14,
        alpha=0.25,
        edgecolors="none",
        zorder=2,
    )

    ax.scatter(
        truth_x0,
        truth_x1,
        color=colors["truth"],
        s=500,
        marker="*",
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.30, linestyle="--", linewidth=0.5)

    ax.set_title(
        rf"{panel_label} $t={time_step}$",
        fontsize=fontsize,
        fontweight="bold",
        pad=4,
    )

    if show_xlabel:
        ax.set_xlabel(r"$x_0$", fontsize=fontsize)
    else:
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel(r"$x_1$", fontsize=fontsize)
    else:
        ax.set_ylabel("")

    ax.tick_params(axis="both", labelsize=fontsize - 2)


# ===========================
# 5. Create 2 x 5 figure
# ===========================

fig, axes = plt.subplots(
    2,
    5,
    figsize=(23, 12),
    sharex=True,
    sharey=True,
)

panel_labels = [
    "(a)", "(b)", "(c)", "(d)", "(e)",
    "(f)", "(g)", "(h)", "(i)", "(j)"
]

for col, t in enumerate(time_steps):
    truth_x0 = x_truth[t, 0]
    truth_x1 = x_truth[t, 1]

    # Top row: EnSF-LR
    plot_single_panel(
        ax=axes[0, col],
        forecast_x0=ensf_lr_x0_forecast[t],
        forecast_x1=ensf_lr_x1_forecast[t],
        analysis_x0=ensf_lr_x0_analysis[t],
        analysis_x1=ensf_lr_x1_analysis[t],
        truth_x0=truth_x0,
        truth_x1=truth_x1,
        time_step=t,
        panel_label=panel_labels[col],
        show_xlabel=False,
        show_ylabel=False,
    )

    # Bottom row: EnKF
    plot_single_panel(
        ax=axes[1, col],
        forecast_x0=enkf_x0_forecast[t],
        forecast_x1=enkf_x1_forecast[t],
        analysis_x0=enkf_x0_analysis[t],
        analysis_x1=enkf_x1_analysis[t],
        truth_x0=truth_x0,
        truth_x1=truth_x1,
        time_step=t,
        panel_label=panel_labels[col + 5],
        show_xlabel=False,
        show_ylabel=False,
    )


# Remove x-axis tick labels from the top row
for ax in axes[0, :]:
    ax.tick_params(labelbottom=False)


# Row labels
fig.text(
    0.025, 0.75, "EnSF-LR",
    rotation=90,
    va="center",
    ha="center",
    fontsize=fontsize + 7,
    fontweight="bold",
)

fig.text(
    0.025, 0.30, "EnKF",
    rotation=90,
    va="center",
    ha="center",
    fontsize=fontsize + 7,
    fontweight="bold",
)


# ===========================
# 6. Shared legend and layout
# ===========================

legend_elements = [
    Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor=colors["forecast"],
        markeredgecolor="none",
        markersize=12,
        alpha=0.25,
        label="Forecast ensemble",
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor=colors["analysis"],
        markeredgecolor="none",
        markersize=12,
        alpha=0.25,
        label="Analysis ensemble",
    ),
    Line2D(
        [0], [0],
        marker="*",
        color="none",
        markerfacecolor=colors["truth"],
        markeredgecolor="black",
        markersize=17,
        label="Reference truth",
    ),
]

fig.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=3,
    fontsize=fontsize + 7 ,
    frameon=True,
)

plt.subplots_adjust(
    left=0.055,
    right=0.998,
    bottom=0.10,
    top=0.94,
    wspace=0.025,
    hspace=0.10,
)

save_path = "figure_EnSF_LR_EnKF_comparison_2x5_scatter_seed30.png"
plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.02
)
#plt.show()

print(f"Figure saved to {save_path}")