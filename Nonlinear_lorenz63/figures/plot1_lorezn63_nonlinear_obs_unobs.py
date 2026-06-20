import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_rmse(filename):
    """Load RMSE values from the second column of a CSV file."""
    return pd.read_csv(filename).iloc[:, 1].to_numpy()


def load_all_seed_curves(filename_pattern, seed_list):
    """
    Load RMSE files over a specified seed list and return an array
    of shape (n_seeds, n_time_steps).
    """
    data = []

    for seed in seed_list:
        filename = filename_pattern.format(seed=seed)
        data.append(load_rmse(filename))

    min_len = min(len(arr) for arr in data)
    data = np.vstack([arr[:min_len] for arr in data])

    return data

# ============================================================
# Seed list
# ============================================================
seed_list = list(range(20))


# ============================================================
# Seed list
# ============================================================
seed_list = list(range(20))


# ============================================================
# Load Lorenz-63 nonlinear observed-state RMSE
# ============================================================
obs_ensf_all = load_all_seed_curves(
    "RMSE_lorenz63_nonlinear_hristo_EnSF_observed_obs_x_10obsgap_seed{seed}.csv",
    seed_list
)

obs_ensflr_all = load_all_seed_curves(
    "RMSE_lorenz63_nonlinear_hristo_EnSF_LR_observed_obs_x_10obsgap_seed{seed}.csv",
    seed_list
)

obs_enkf_all = load_all_seed_curves(
    "RMSE_lorenz63_nonlinear_hristo_EnKF_observed_obs_x_10obsgap_seed{seed}.csv",
    seed_list
)


# ============================================================
# Load Lorenz-63 nonlinear unobserved-state RMSE
# ============================================================
unobs_ensf_all = load_all_seed_curves(
    "RMSE_lorenz63_nonlinear_hristo_EnSF_unobserved_obs_x_10obsgap_seed{seed}.csv",
    seed_list
)

unobs_ensflr_all = load_all_seed_curves(
    "RMSE_lorenz63_nonlinear_hristo_EnSF_LR_unobserved_obs_x_10obsgap_seed{seed}.csv",
    seed_list
)

unobs_enkf_all = load_all_seed_curves(
    "RMSE_lorenz63_nonlinear_hristo_EnKF_unobserved_obs_x_10obsgap_seed{seed}.csv",
    seed_list
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


obs_ensf_mean, obs_ensf_lo, obs_ensf_hi = summarize_curves_mean_ci(obs_ensf_all)
obs_ensflr_mean, obs_ensflr_lo, obs_ensflr_hi = summarize_curves_mean_ci(obs_ensflr_all)
obs_enkf_mean, obs_enkf_lo, obs_enkf_hi = summarize_curves_mean_ci(obs_enkf_all)

unobs_ensf_mean, unobs_ensf_lo, unobs_ensf_hi = summarize_curves_mean_ci(unobs_ensf_all)
unobs_ensflr_mean, unobs_ensflr_lo, unobs_ensflr_hi = summarize_curves_mean_ci(unobs_ensflr_all)
unobs_enkf_mean, unobs_enkf_lo, unobs_enkf_hi = summarize_curves_mean_ci(unobs_enkf_all)


mean_obs_ensf = np.mean(obs_ensf_mean[-20:])
mean_obs_ensflr = np.mean(obs_ensflr_mean[-20:])
mean_obs_enkf = np.mean(obs_enkf_mean[-20:])

mean_unobs_ensf = np.mean(unobs_ensf_mean[-20:])
mean_unobs_ensflr = np.mean(unobs_ensflr_mean[-20:])
mean_unobs_enkf = np.mean(unobs_enkf_mean[-20:])


fontsize = 12

fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

# -------------------------
# Panel (a): Observed state
# -------------------------
line1, = axs[0].plot(x_analysis, obs_ensf_mean, linewidth=2, marker="o", markersize=4, label="EnSF")
line2, = axs[0].plot(x_analysis, obs_ensflr_mean, linewidth=2, marker="o", markersize=4, label="EnSF-LR")
line3, = axs[0].plot(x_analysis, obs_enkf_mean, linewidth=2, marker="o", markersize=4, label="EnKF")

axs[0].fill_between(x_analysis, obs_ensf_lo, obs_ensf_hi, color=line1.get_color(), alpha=0.05)
axs[0].fill_between(x_analysis, obs_ensflr_lo, obs_ensflr_hi, color=line2.get_color(), alpha=0.2)
axs[0].fill_between(x_analysis, obs_enkf_lo, obs_enkf_hi, color=line3.get_color(), alpha=0.2)

axs[0].set_title("(a) Observed state", fontsize=fontsize + 2)
axs[0].set_xlabel("Filtering step", fontsize=fontsize + 2)
axs[0].set_ylabel("Observed-state RMSE", fontsize=fontsize + 2)
axs[0].tick_params(axis="x", labelsize=fontsize + 1)
axs[0].tick_params(axis="y", labelsize=fontsize + 1)
axs[0].grid(True, alpha=0.3)

# -------------------------
# Panel (b): Unobserved states
# -------------------------
line1_b, = axs[1].plot(x_analysis, unobs_ensf_mean, linewidth=2, marker="o", markersize=4, label="EnSF")
line2_b, = axs[1].plot(x_analysis, unobs_ensflr_mean, linewidth=2, marker="o", markersize=4, label="EnSF-LR")
line3_b, = axs[1].plot(x_analysis, unobs_enkf_mean, linewidth=2, marker="o", markersize=4, label="EnKF")

axs[1].fill_between(x_analysis, unobs_ensf_lo, unobs_ensf_hi, color=line1_b.get_color(), alpha=0.05)
axs[1].fill_between(x_analysis, unobs_ensflr_lo, unobs_ensflr_hi, color=line2_b.get_color(), alpha=0.2)
axs[1].fill_between(x_analysis, unobs_enkf_lo, unobs_enkf_hi, color=line3_b.get_color(), alpha=0.2)

axs[1].set_title("(b) Unobserved states", fontsize=fontsize + 2)
axs[1].set_xlabel("Filtering step", fontsize=fontsize + 2)
axs[1].set_ylabel("Unobserved-state RMSE", fontsize=fontsize + 2)
axs[1].tick_params(axis="x", labelsize=fontsize + 1)
axs[1].tick_params(axis="y", labelsize=fontsize + 1)
axs[1].grid(True, alpha=0.3)



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
    "figure_lorenz63_nonlinear_observed_unobserved_analysis_only_with_spread.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()