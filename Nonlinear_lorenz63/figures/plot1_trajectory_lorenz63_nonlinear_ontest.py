import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


def plot_lorenz63_xyz_ensemble_comparison(
    x_truth,
    enkf_ensembles,
    ensflr_ensembles,
    variable_names=(r"Observed state $X$", r"Unobserved state $Y$", r"Unobserved state $Z$"),
    method_names=("EnKF", "EnSF-LR"),
    time_steps=None,
    obs_gap=10,
    n_ensemble_curves=1000,
    seed=42,
    save_path=None,
    ylims=None,
    fontsize=12
):
    """
    Plot Lorenz-63 X, Y, and Z trajectory comparisons in one large figure.

    Rows:
        X, Y, Z

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

    fig, axes = plt.subplots(
        3, 2,
        figsize=(15, 11),
        sharex=True,
        sharey='row',
        constrained_layout=False
    )

    da_indices = np.arange(0, len(time_steps), obs_gap)
    da_time_values = time_steps[da_indices]

    for row in range(3):
        truth_var = x_truth[:, row]

        for col, (method_name, ensemble_dict) in enumerate(
            zip(method_names, [enkf_ensembles, ensflr_ensembles])
        ):
            ax = axes[row, col]
            ensemble = np.asarray(ensemble_dict[row])

            T, N_ens = ensemble.shape
            assert len(truth_var) == T, "Truth and ensemble time lengths do not match."

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
                    alpha=0.03,
                    linewidth=0.3,
                    label="_nolegend_"
                )

            # Ensemble mean
            ax.plot(
                time_steps,
                ensemble_mean,
                color=COLOR_MEAN,
                linewidth=2.4,
                label="Ensemble mean",
                zorder=10
            )

            # Reference truth
            ax.plot(
                time_steps,
                truth_var,
                color=COLOR_TRUTH,
                linewidth=2.4,
                label="Reference truth",
                zorder=11
            )

            # DA update lines
            for t in da_time_values:
                ax.axvline(
                    x=t,
                    color="gray",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.25,
                    label="_nolegend_"
                )

            ax.grid(True, alpha=0.3, linestyle="--")
            ax.tick_params(axis="both", labelsize=fontsize + 1)

            if row == 0:
                ax.set_title(method_name, fontsize=fontsize + 2)

            if col == 0:
                ax.set_ylabel(variable_names[row], fontsize=fontsize + 1, labelpad=18)
                ax.yaxis.set_label_coords(-0.08, 0.5)

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
        label="Ensemble members"
    )

    mean_marker = Line2D(
        [0], [0],
        color=COLOR_MEAN,
        linewidth=2.4,
        label="Ensemble mean"
    )

    truth_marker = Line2D(
        [0], [0],
        color=COLOR_TRUTH,
        linewidth=2.4,
        label="Reference truth"
    )

    fig.legend(
        handles=[ensemble_marker, mean_marker, truth_marker],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        fontsize=fontsize+2,
        frameon=True
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()

    return fig, axes


# ============================================================
# Load truth
# ============================================================
x_truth = pd.read_csv(
    "x_truth_trajectory_lorenz63_seed4.csv"
).iloc[:, 1:].to_numpy().copy()   # shape: (300, 3)


# ============================================================
# Load EnKF ensemble trajectories
# ============================================================
x_ensemble_enkf_x = pd.read_csv(
    "x_0_traj_nonlinear_hristo_EnKF_LR_lorenz63_33%obs_10obsgap_seed4.csv"
).iloc[:300, 1:].to_numpy().copy()

x_ensemble_enkf_y = pd.read_csv(
    "x_1_traj_nonlinear_hristo_EnKF_LR_lorenz63_33%obs_10obsgap_seed4.csv"
).iloc[:300, 1:].to_numpy().copy()

x_ensemble_enkf_z = pd.read_csv(
    "x_2_traj_nonlinear_hristo_EnKF_LR_lorenz63_33%obs_10obsgap_seed4.csv"
).iloc[:300, 1:].to_numpy().copy()


# ============================================================
# Load EnSF-LR ensemble trajectories
# ============================================================
x_ensemble_ensf_lr_x = pd.read_csv(
    "x_0_traj_nonlinear_hristo_EnSF_LR_lorenz63_33%obs_10obsgap_seed4.csv"
).iloc[:300, 1:].to_numpy().copy()

x_ensemble_ensf_lr_y = pd.read_csv(
    "x_1_traj_nonlinear_hristo_EnSF_LR_lorenz63_33%obs_10obsgap_seed4.csv"
).iloc[:300, 1:].to_numpy().copy()

x_ensemble_ensf_lr_z = pd.read_csv(
    "x_2_traj_nonlinear_hristo_EnSF_LR_lorenz63_33%obs_10obsgap_seed4.csv"
).iloc[:300, 1:].to_numpy().copy()


# Store variables in order: X, Y, Z
enkf_ensembles = {
    0: x_ensemble_enkf_x,
    1: x_ensemble_enkf_y,
    2: x_ensemble_enkf_z,
}

ensflr_ensembles = {
    0: x_ensemble_ensf_lr_x,
    1: x_ensemble_ensf_lr_y,
    2: x_ensemble_ensf_lr_z,
}


fig, axes = plot_lorenz63_xyz_ensemble_comparison(
    x_truth=x_truth,
    enkf_ensembles=enkf_ensembles,
    ensflr_ensembles=ensflr_ensembles,
    variable_names=(r"Observed state $X$", r"Unobserved state $Y$", r"Unobserved state $Z$"),
    time_steps=np.arange(300),
    obs_gap=10,
    n_ensemble_curves=1000,
    seed=42,
    ylims=[(-25, 25), (-35, 35), (0, 55)],
    fontsize=15,
    save_path="figure_trajectory_lorenz63_xyz_EnKF_vs_EnSF_LR_nonlinear_33obs_10obsgap_seed4.png"
)