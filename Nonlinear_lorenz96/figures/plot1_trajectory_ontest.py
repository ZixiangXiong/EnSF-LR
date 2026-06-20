import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


def plot_lorenz96_selected_states_ensemble_comparison(
    x_truth,
    enkf_ensembles,
    ensflr_ensembles,
    variable_indices,
    variable_names,
    method_names=("EnKF", "EnSF-LR"),
    time_steps=None,
    obs_gap=10,
    n_ensemble_curves=2000,
    seed=42,
    ylims=None,
    save_path=None,
    fontsize=15,
):
    """
    Plot selected Lorenz-96 state trajectory comparisons in one large figure.

    Rows:
        selected state variables, e.g., x_0 and x_1

    Columns:
        EnKF, EnSF-LR
    """

    x_truth = np.asarray(x_truth)

    if time_steps is None:
        time_steps = np.arange(x_truth.shape[0])

    COLOR_ENSEMBLE = "#808080"
    COLOR_MEAN = "#0072B2"
    COLOR_TRUTH = "#D55E00"

    rng = np.random.default_rng(seed)
    n_rows = len(variable_indices)

    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(15, 4.6 * n_rows),
        sharex=True,
        sharey="row",
        constrained_layout=False,
    )

    if n_rows == 1:
        axes = np.array([axes])

    da_indices = np.arange(0, len(time_steps), obs_gap)
    da_time_values = time_steps[da_indices]

    for row, var_idx in enumerate(variable_indices):
        truth_var = x_truth[:, var_idx]

        for col, (method_name, ensemble_dict) in enumerate(
            zip(method_names, [enkf_ensembles, ensflr_ensembles])
        ):
            ax = axes[row, col]
            ensemble = np.asarray(ensemble_dict[var_idx])

            T, N_ens = ensemble.shape

            if len(truth_var) != T:
                raise ValueError("Truth and ensemble time lengths do not match.")

            if n_ensemble_curves >= N_ens:
                selected = np.arange(N_ens)
            else:
                selected = rng.choice(N_ens, size=n_ensemble_curves, replace=False)

            ensemble_mean = np.mean(ensemble, axis=1)

            # Ensemble members
            for idx in selected:
                ax.plot(
                    time_steps,
                    ensemble[:, idx],
                    color=COLOR_ENSEMBLE,
                    alpha=0.025,
                    linewidth=0.3,
                    label="_nolegend_",
                )

            # Ensemble mean
            ax.plot(
                time_steps,
                ensemble_mean,
                color=COLOR_MEAN,
                linewidth=2.4,
                label="Ensemble mean",
                zorder=10,
            )

            # Reference truth
            ax.plot(
                time_steps,
                truth_var,
                color=COLOR_TRUTH,
                linewidth=2.4,
                label="Reference truth",
                zorder=11,
            )

            # DA update lines
            for t in da_time_values:
                ax.axvline(
                    x=t,
                    color="gray",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.25,
                    label="_nolegend_",
                )

            ax.grid(True, alpha=0.3, linestyle="--")
            ax.tick_params(axis="both", labelsize=fontsize + 1)

            if row == 0:
                ax.set_title(method_name, fontsize=fontsize + 2)

            if col == 0:
                ax.set_ylabel(variable_names[row], fontsize=fontsize + 1, labelpad=18)
                ax.yaxis.set_label_coords(-0.08, 0.5)

            # Remove duplicated y-axis numbers on the right column
            if col == 1:
                ax.tick_params(axis="y", labelleft=False)

            if ylims is not None:
                ax.set_ylim(ylims[row])

    axes[-1, 0].set_xlabel("Filtering step", fontsize=fontsize + 2)
    axes[-1, 1].set_xlabel("Filtering step", fontsize=fontsize + 2)

    # Shared legend
    ensemble_marker = Line2D(
        [0], [0],
        color=COLOR_ENSEMBLE,
        linewidth=1.5,
        alpha=0.6,
        label="Ensemble members",
    )

    mean_marker = Line2D(
        [0], [0],
        color=COLOR_MEAN,
        linewidth=2.4,
        label="Ensemble mean",
    )

    truth_marker = Line2D(
        [0], [0],
        color=COLOR_TRUTH,
        linewidth=2.4,
        label="Reference truth",
    )

    fig.legend(
        handles=[ensemble_marker, mean_marker, truth_marker],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        fontsize=fontsize + 2,
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()

    return fig, axes


# ============================================================
# Load data
# ============================================================

x_truth = pd.read_csv(
    "x_truth_trajectory_seed30.csv"
).iloc[:600, 1:].to_numpy().copy()                              # shape: (600, 40)


# EnKF ensemble trajectories
x_ensemble_enkf_x0 = pd.read_csv(
    "x_0_traj_hristo_EnKF_nonlinear_25%obs_10obsgap_seed30.csv"
).iloc[:600, 1:-1:5].to_numpy().copy()                          # shape: (600, 2000)


x_ensemble_enkf_x1 = pd.read_csv(
    "x_1_traj_hristo_EnKF_nonlinear_25%obs_10obsgap_seed30.csv"
).iloc[:600, 1:-1:5].to_numpy().copy()


# EnSF-LR ensemble trajectories
x_ensemble_ensf_lr_x0 = pd.read_csv(
    "x_0_traj_hristo_EnSF_nonlinear_LR_25%obs_10obsgap_seed30.csv"
).iloc[:600, 1:-1:5].to_numpy().copy()

x_ensemble_ensf_lr_x1 = pd.read_csv(
    "x_1_traj_hristo_EnSF_nonlinear_LR_25%obs_10obsgap_seed30.csv"
).iloc[:600, 1:-1:5].to_numpy().copy()


# Store by state index
enkf_ensembles = {
    0: x_ensemble_enkf_x0,
    1: x_ensemble_enkf_x1,
}

ensflr_ensembles = {
    0: x_ensemble_ensf_lr_x0,
    1: x_ensemble_ensf_lr_x1,
}


# ============================================================
# Plot combined trajectory figure
# ============================================================

fig, axes = plot_lorenz96_selected_states_ensemble_comparison(
    x_truth=x_truth,
    enkf_ensembles=enkf_ensembles,
    ensflr_ensembles=ensflr_ensembles,
    variable_indices=[0, 1],
    variable_names=[
        r"Observed state $x_0$",
        r"Unobserved state $x_1$",
    ],
    time_steps=np.arange(600),
    obs_gap=10,
    n_ensemble_curves=2000,
    seed=42,
    ylims=[
        (-10, 15),
        (-10, 15),
    ],
    fontsize=15,
    save_path="figure_trajectory_lorenz96_x0_x1_EnKF_vs_EnSF_LR_nonlinear_25obs_10obsgap_seed30.png",
)