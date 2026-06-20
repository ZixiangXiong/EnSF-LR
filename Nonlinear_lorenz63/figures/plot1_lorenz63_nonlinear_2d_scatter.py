import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ===========================
# 1. Load data manually
# ===========================

N_TIME = 300
N_ENSEMBLE = 1000

x_truth = pd.read_csv(
    "x_truth_trajectory_lorenz63_seed4.csv"
).iloc[:N_TIME, 1:].to_numpy()

# EnSF-LR outputs
ensf_lr_x_forecast = pd.read_csv(
    "x_0_traj_nonlinear_forecast_hristo_EnSF_LR_lorenz63_33%obs_10obsgap_seed4.csv"
).iloc[:N_TIME, 1:].to_numpy()

ensf_lr_x_analysis = pd.read_csv(
    "x_0_traj_nonlinear_hristo_EnSF_LR_lorenz63_33%obs_10obsgap_seed4.csv"
).iloc[:N_TIME, 1:].to_numpy()

ensf_lr_y_forecast = pd.read_csv(
    "x_1_traj_nonlinear_forecast_hristo_EnSF_LR_lorenz63_33%obs_10obsgap_seed4.csv"
).iloc[:N_TIME, 1:].to_numpy()

ensf_lr_y_analysis = pd.read_csv(
    "x_1_traj_nonlinear_hristo_EnSF_LR_lorenz63_33%obs_10obsgap_seed4.csv"
).iloc[:N_TIME, 1:].to_numpy()

# EnKF outputs
enkf_x_forecast = pd.read_csv(
    "x_0_traj_nonlinear_forecast_hristo_EnKF_LR_lorenz63_33%obs_10obsgap_seed4.csv"
).iloc[:N_TIME, 1:].to_numpy()

enkf_x_analysis = pd.read_csv(
    "x_0_traj_nonlinear_hristo_EnKF_LR_lorenz63_33%obs_10obsgap_seed4.csv"
).iloc[:N_TIME, 1:].to_numpy()

enkf_y_forecast = pd.read_csv(
    "x_1_traj_nonlinear_forecast_hristo_EnKF_LR_lorenz63_33%obs_10obsgap_seed4.csv"
).iloc[:N_TIME, 1:].to_numpy()

enkf_y_analysis = pd.read_csv(
    "x_1_traj_nonlinear_hristo_EnKF_LR_lorenz63_33%obs_10obsgap_seed4.csv"
).iloc[:N_TIME, 1:].to_numpy()


# ===========================
# 2. Configuration
# ===========================

time_steps = [70, 170, 270]
subsample_size = 300

rng = np.random.default_rng(42)
subsample_idx = rng.choice(N_ENSEMBLE, size=subsample_size, replace=False)

colors = {
    "forecast": "#9ecae1",   # light blue
    "analysis": "#d62728",   # red
    "truth": "#ffd700", # gold
    #
}


# ===========================
# 3. Compute shared axis limits
# ===========================

all_x = []
all_y = []

for t in time_steps:
    # EnSF-LR
    all_x.extend(ensf_lr_x_forecast[t, subsample_idx])
    all_x.extend(ensf_lr_x_analysis[t, subsample_idx])
    all_y.extend(ensf_lr_y_forecast[t, subsample_idx])
    all_y.extend(ensf_lr_y_analysis[t, subsample_idx])

    # EnKF
    all_x.extend(enkf_x_forecast[t, subsample_idx])
    all_x.extend(enkf_x_analysis[t, subsample_idx])
    all_y.extend(enkf_y_forecast[t, subsample_idx])
    all_y.extend(enkf_y_analysis[t, subsample_idx])

    # Truth
    all_x.append(x_truth[t, 0])
    all_y.append(x_truth[t, 1])

all_x = np.asarray(all_x)
all_y = np.asarray(all_y)

margin = 0.15
x_min, x_max = np.nanmin(all_x), np.nanmax(all_x)
y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)

x_range = x_max - x_min
y_range = y_max - y_min

xlim = (x_min - margin * x_range, x_max + margin * x_range)
ylim = (y_min - margin * y_range, y_max + margin * y_range)


# ===========================
# 4. Plot helper
# ===========================

def plot_single_panel(
    ax,
    forecast_x,
    forecast_y,
    analysis_x,
    analysis_y,
    truth_x,
    truth_y,
    time_step,
    panel_label,
    show_xlabel=False,
    show_ylabel=False,
):
    """Plot forecast ensemble, analysis ensemble, and truth in the X-Y plane."""

    ax.scatter(
        forecast_x[subsample_idx],
        forecast_y[subsample_idx],
        color=colors["forecast"],
        s=12,
        alpha=0.25,
        edgecolors="none",
        zorder=1,
    )

    ax.scatter(
        analysis_x[subsample_idx],
        analysis_y[subsample_idx],
        color=colors["analysis"],
        s=12,
        alpha=0.38,
        edgecolors="none",
        zorder=2,
    )

    ax.scatter(
        truth_x,
        truth_y,
        color=colors["truth"],
        s=150,
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
        fontsize=13,
        fontweight="bold",
        pad=8,
    )

    if show_xlabel:
        ax.set_xlabel(r"$X$", fontsize=13)
    else:
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel(r"$Y$", fontsize=13)
    else:
        ax.set_ylabel("")

    ax.tick_params(axis="both", labelsize=11)


# ===========================
# 5. Create 2 x 3 figure
# ===========================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 8.5),
    sharex=True,
    sharey=True,
)

panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

for col, t in enumerate(time_steps):
    truth_x = x_truth[t, 0]
    truth_y = x_truth[t, 1]

    # Top row: EnSF-LR
    plot_single_panel(
        ax=axes[0, col],
        forecast_x=ensf_lr_x_forecast[t],
        forecast_y=ensf_lr_y_forecast[t],
        analysis_x=ensf_lr_x_analysis[t],
        analysis_y=ensf_lr_y_analysis[t],
        truth_x=truth_x,
        truth_y=truth_y,
        time_step=t,
        panel_label=panel_labels[col],
        show_xlabel=False,
        show_ylabel=(col == 0),
    )

    # Bottom row: EnKF
    plot_single_panel(
        ax=axes[1, col],
        forecast_x=enkf_x_forecast[t],
        forecast_y=enkf_y_forecast[t],
        analysis_x=enkf_x_analysis[t],
        analysis_y=enkf_y_analysis[t],
        truth_x=truth_x,
        truth_y=truth_y,
        time_step=t,
        panel_label=panel_labels[col + 3],
        show_xlabel=True,
        show_ylabel=(col == 0),
    )
fig.text(
    0.04, 0.75, "EnSF-LR",
    rotation=90,
    va="center",
    ha="center",
    fontsize=16,
    fontweight="bold"
)

fig.text(
    0.04, 0.30, "EnKF",
    rotation=90,
    va="center",
    ha="center",
    fontsize=16,
    fontweight="bold"
)

# ===========================
# 6. Legend and layout
# ===========================

legend_elements = [
    Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor=colors["forecast"],
        markeredgecolor="none",
        markersize=8,
        alpha=0.35,
        label="Forecast ensemble",
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor=colors["analysis"],
        markeredgecolor="none",
        markersize=8,
        alpha=0.55,
        label="Analysis ensemble",
    ),
    Line2D(
        [0], [0],
        marker="*",
        color="none",
        markerfacecolor=colors["truth"],
        markeredgecolor="black",
        markersize=13,
        label="Reference truth",
    ),
]

fig.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.55, -0.01),
    ncol=3,
    fontsize=16,
    frameon=True,
)


plt.subplots_adjust(
    left=0.11,
    right=0.98,
    bottom=0.12,
    top=0.92,
    wspace=0.15,
    hspace=0.25,
)

save_path = "figure_2x3_scatter_lorenz63_EnSF_LR_EnKF_comparison_nonlinear_seed4.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"✓ Figure saved to {save_path}")