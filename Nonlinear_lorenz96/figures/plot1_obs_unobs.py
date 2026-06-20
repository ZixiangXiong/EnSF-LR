import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_rmse(filename):
    """Load RMSE values from the second column of a CSV file."""
    return pd.read_csv(filename).iloc[:, 1].to_numpy()


def load_all_seed_curves(filename_pattern, seed_list, max_steps=None):
    """
    Load RMSE files over a specified seed list and return an array
    of shape (n_seeds, n_time_steps).

    If max_steps is not None, only keep the first max_steps time steps.
    """
    data = []

    for seed in seed_list:
        filename = filename_pattern.format(seed=seed)
        curve = load_rmse(filename)

        if max_steps is not None:
            curve = curve[:max_steps]

        data.append(curve)

    min_len = min(len(arr) for arr in data)
    data = np.vstack([arr[:min_len] for arr in data])

    return data


def summarize_curves_mean_ci(curves):
    """
    curves shape: (n_seeds, n_times)
    Returns mean curve and 95% confidence interval for the mean.
    """
    n = curves.shape[0]
    mean = np.mean(curves, axis=0)
    std = np.std(curves, axis=0, ddof=1)

    # t_{0.975,19} for 20 realizations
    tcrit = 2.093
    half_width = tcrit * std / np.sqrt(n)

    lower = mean - half_width
    upper = mean + half_width

    return mean, lower, upper


# ============================================================
# Seed list
# ============================================================
seed_list = list(range(20,40))


# ============================================================
# Load Lorenz-96 nonlinear observed-state RMSE
# ============================================================
MAX_STEPS = 600

obs_ensf_all = load_all_seed_curves(
    "RMSE_hristo_EnSF_nonlinear_observed_25%obs_10obsgap_seed{seed}.csv",
    seed_list, max_steps=MAX_STEPS
)

obs_ensflr_all = load_all_seed_curves(
    "RMSE_hristo_EnSF_nonlinear_LR_observed_25%obs_10obsgap_seed{seed}.csv",
    seed_list, max_steps=MAX_STEPS
)

obs_enkf_all = load_all_seed_curves(
    "RMSE_hristo_EnKF_nonlinear_observed_25%obs_10obsgap_seed{seed}.csv",
    seed_list, max_steps=MAX_STEPS
)


# ============================================================
# Load Lorenz-96 nonlinear unobserved-state RMSE
# ============================================================
unobs_ensf_all = load_all_seed_curves(
    "RMSE_hristo_EnSF_nonlinear_unobserved_25%obs_10obsgap_seed{seed}.csv",
    seed_list, max_steps=MAX_STEPS
)

unobs_ensflr_all = load_all_seed_curves(
    "RMSE_hristo_EnSF_nonlinear_LR_unobserved_25%obs_10obsgap_seed{seed}.csv",
    seed_list, max_steps=MAX_STEPS
)

unobs_enkf_all = load_all_seed_curves(
    "RMSE_hristo_EnKF_nonlinear_unobserved_25%obs_10obsgap_seed{seed}.csv",
    seed_list, max_steps=MAX_STEPS
)


# ============================================================
# Keep only analysis times
# ============================================================
obs_gap = 10


analysis_idx = np.arange(0, obs_ensf_all.shape[1], obs_gap)
x_analysis = analysis_idx

obs_ensf_all = obs_ensf_all[:, analysis_idx]
obs_ensflr_all = obs_ensflr_all[:, analysis_idx]
obs_enkf_all = obs_enkf_all[:, analysis_idx]

unobs_ensf_all = unobs_ensf_all[:, analysis_idx]
unobs_ensflr_all = unobs_ensflr_all[:, analysis_idx]
unobs_enkf_all = unobs_enkf_all[:, analysis_idx]


# ============================================================
# Mean curves and 95% CI bands
# ============================================================
obs_ensf_mean, obs_ensf_lo, obs_ensf_hi = summarize_curves_mean_ci(obs_ensf_all)
obs_ensflr_mean, obs_ensflr_lo, obs_ensflr_hi = summarize_curves_mean_ci(obs_ensflr_all)
obs_enkf_mean, obs_enkf_lo, obs_enkf_hi = summarize_curves_mean_ci(obs_enkf_all)

unobs_ensf_mean, unobs_ensf_lo, unobs_ensf_hi = summarize_curves_mean_ci(unobs_ensf_all)
unobs_ensflr_mean, unobs_ensflr_lo, unobs_ensflr_hi = summarize_curves_mean_ci(unobs_ensflr_all)
unobs_enkf_mean, unobs_enkf_lo, unobs_enkf_hi = summarize_curves_mean_ci(unobs_enkf_all)


# ============================================================
# Plot combined figure
# ============================================================
fontsize = 12

fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

# -------------------------
# Panel (a): Observed states
# -------------------------
line1, = axs[0].plot(
    x_analysis, obs_ensf_mean,
    linewidth=2,
    marker="o",
    markersize=4,
    label="EnSF"
)

line2, = axs[0].plot(
    x_analysis, obs_ensflr_mean,
    linewidth=2,
    marker="o",
    markersize=4,
    label="EnSF-LR"
)

line3, = axs[0].plot(
    x_analysis, obs_enkf_mean,
    linewidth=2,
    marker="o",
    markersize=4,
    label="EnKF"
)

axs[0].fill_between(
    x_analysis, obs_ensf_lo, obs_ensf_hi,
    color=line1.get_color(),
    alpha=0.05
)

axs[0].fill_between(
    x_analysis, obs_ensflr_lo, obs_ensflr_hi,
    color=line2.get_color(),
    alpha=0.20
)

axs[0].fill_between(
    x_analysis, obs_enkf_lo, obs_enkf_hi,
    color=line3.get_color(),
    alpha=0.20
)

axs[0].set_title("(a) Observed states", fontsize=fontsize + 2)
axs[0].set_xlabel("Filtering step", fontsize=fontsize + 2)
axs[0].set_ylabel("Observed-state RMSE", fontsize=fontsize + 2)
axs[0].tick_params(axis="x", labelsize=fontsize + 1)
axs[0].tick_params(axis="y", labelsize=fontsize + 1)
axs[0].grid(True, alpha=0.3)


# -------------------------
# Panel (b): Unobserved states
# -------------------------
line1_b, = axs[1].plot(
    x_analysis, unobs_ensf_mean,
    linewidth=2,
    marker="o",
    markersize=4,
    label="EnSF"
)

line2_b, = axs[1].plot(
    x_analysis, unobs_ensflr_mean,
    linewidth=2,
    marker="o",
    markersize=4,
    label="EnSF-LR"
)

line3_b, = axs[1].plot(
    x_analysis, unobs_enkf_mean,
    linewidth=2,
    marker="o",
    markersize=4,
    label="EnKF"
)

axs[1].fill_between(
    x_analysis, unobs_ensf_lo, unobs_ensf_hi,
    color=line1_b.get_color(),
    alpha=0.05
)

axs[1].fill_between(
    x_analysis, unobs_ensflr_lo, unobs_ensflr_hi,
    color=line2_b.get_color(),
    alpha=0.20
)

axs[1].fill_between(
    x_analysis, unobs_enkf_lo, unobs_enkf_hi,
    color=line3_b.get_color(),
    alpha=0.20
)

axs[1].set_title("(b) Unobserved states", fontsize=fontsize + 2)
axs[1].set_xlabel("Filtering step", fontsize=fontsize + 2)
axs[1].set_ylabel("Unobserved-state RMSE", fontsize=fontsize + 2)
axs[1].tick_params(axis="x", labelsize=fontsize + 1)
axs[1].tick_params(axis="y", labelsize=fontsize + 1)
axs[1].grid(True, alpha=0.3)


# ============================================================
# Shared legend
# ============================================================
plt.tight_layout(rect=[0, 0.14, 1, 1])

fig.legend(
    handles=[line1, line2, line3],
    labels=["EnSF", "EnSF-LR", "EnKF"],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.05),
    ncol=3,
    fontsize=fontsize + 1,
    frameon=True,
)


plt.savefig(
    "figure_lorenz96_nonlinear_observed_unobserved_analysis_only_with_spread.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()