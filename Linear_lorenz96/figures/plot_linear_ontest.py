import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_rmse(filename):
    """Load RMSE values from the second column of a CSV file."""
    return pd.read_csv(filename).iloc[:, 1].to_numpy()


def load_all_seed_curves(filename_pattern, seed_list, max_steps=None):
    """
    Load RMSE files over a specified seed list.

    Returns
    -------
    np.ndarray
        Array with shape (n_seeds, n_time_steps).
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


def keep_da_steps(curves, obs_gap=10):
    """
    Keep only RMSE values at DA analysis time steps.

    Parameters
    ----------
    curves : np.ndarray
        Shape (n_seeds, n_time_steps).

    Returns
    -------
    da_idx : np.ndarray
        DA time-step indices.
    curves_da : np.ndarray
        Shape (n_seeds, n_da_steps).
    """
    n_time = curves.shape[1]

    # Keep your current convention: DA analysis steps are 1, 11, 21, ...
    da_idx = np.arange(1, n_time, obs_gap)

    return da_idx, curves[:, da_idx]


def mean_and_ci(curves_da):
    """
    Compute seed mean and pointwise 95% confidence interval.

    Parameters
    ----------
    curves_da : np.ndarray
        Shape (n_seeds, n_da_steps).
    """
    n_seed = curves_da.shape[0]

    mean_curve = np.mean(curves_da, axis=0)
    std_curve = np.std(curves_da, axis=0, ddof=1)

    # t_{0.975, 19} = 2.093 for 20 seeds
    tcrit = 2.093
    half_width = tcrit * std_curve / np.sqrt(n_seed)

    lower = mean_curve - half_width
    upper = mean_curve + half_width

    return mean_curve, lower, upper


def mean_over_last_steps_at_da(x_da, curves_da, last_steps=200):
    """
    Compute mean RMSE over DA analysis steps within the last `last_steps`
    filtering steps.

    First compute the DA-time mean for each seed, then average over seeds.
    """
    start_eval = x_da[-1] - last_steps + 1
    eval_mask = x_da >= start_eval

    seed_means = np.mean(curves_da[:, eval_mask], axis=1)

    return np.mean(seed_means)


# ============================================================
# Settings
# ============================================================
obs_gap = 10
last_steps = 200

l63_seed_list = list(range(20))

# Lorenz-96: intentionally use seed 1 twice and skip seed 2
l96_seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

alpha_band = 0.15
# ============================================================
# Load Lorenz-63 linear-observation results
# ============================================================
l63_ensf_all = load_all_seed_curves(
    "RMSE_lorenz63_linear_hristo_EnSF_obs_x_10obsgap_seed{seed}.csv",
    l63_seed_list,
)

l63_ensflr_all = load_all_seed_curves(
    "RMSE_lorenz63_linear_hristo_EnSF_LR_obs_x_10obsgap_seed{seed}.csv",
    l63_seed_list,
)

l63_enkf_all = load_all_seed_curves(
    "RMSE_lorenz63_linear_hristo_EnKF_obs_x_10obsgap_seed{seed}.csv",
    l63_seed_list,
)


# ============================================================
# Load Lorenz-96 linear-observation results
# ============================================================
l96_ensf_all = load_all_seed_curves(
    "RMSE_lorenz96_linear_hristo_EnSF_25%obs_10obsgap_seed{seed}.csv",
    l96_seed_list,max_steps=600
)

l96_ensflr_all = load_all_seed_curves(
    "RMSE_lorenz96_linear_hristo_EnSF_LR_25%obs_10obsgap_seed{seed}.csv",
    l96_seed_list,max_steps=600
)

l96_enkf_all = load_all_seed_curves(
    "RMSE_lorenz96_linear_hristo_EnKF_25%obs_10obsgap_seed{seed}.csv",
    l96_seed_list,max_steps=600
)


# ============================================================
# Keep only DA analysis-time RMSE values
# ============================================================
x_l63, l63_ensf_da_all = keep_da_steps(l63_ensf_all, obs_gap=obs_gap)
_, l63_ensflr_da_all = keep_da_steps(l63_ensflr_all, obs_gap=obs_gap)
_, l63_enkf_da_all = keep_da_steps(l63_enkf_all, obs_gap=obs_gap)

x_l96, l96_ensf_da_all = keep_da_steps(l96_ensf_all, obs_gap=obs_gap)
_, l96_ensflr_da_all = keep_da_steps(l96_ensflr_all, obs_gap=obs_gap)
_, l96_enkf_da_all = keep_da_steps(l96_enkf_all, obs_gap=obs_gap)


# ============================================================
# Compute mean curves and confidence bands
# ============================================================
l63_ensf_mean, l63_ensf_low, l63_ensf_high = mean_and_ci(l63_ensf_da_all)
l63_ensflr_mean, l63_ensflr_low, l63_ensflr_high = mean_and_ci(l63_ensflr_da_all)
l63_enkf_mean, l63_enkf_low, l63_enkf_high = mean_and_ci(l63_enkf_da_all)

l96_ensf_mean, l96_ensf_low, l96_ensf_high = mean_and_ci(l96_ensf_da_all)
l96_ensflr_mean, l96_ensflr_low, l96_ensflr_high = mean_and_ci(l96_ensflr_da_all)
l96_enkf_mean, l96_enkf_low, l96_enkf_high = mean_and_ci(l96_enkf_da_all)


# ============================================================
# Compute mean RMSE over DA steps in the last 200 filtering steps
# ============================================================
mean_l63_ensf = mean_over_last_steps_at_da(x_l63, l63_ensf_da_all, last_steps=last_steps)
mean_l63_ensflr = mean_over_last_steps_at_da(x_l63, l63_ensflr_da_all, last_steps=last_steps)
mean_l63_enkf = mean_over_last_steps_at_da(x_l63, l63_enkf_da_all, last_steps=last_steps)

mean_l96_ensf = mean_over_last_steps_at_da(x_l96, l96_ensf_da_all, last_steps=last_steps)
mean_l96_ensflr = mean_over_last_steps_at_da(x_l96, l96_ensflr_da_all, last_steps=last_steps)
mean_l96_enkf = mean_over_last_steps_at_da(x_l96, l96_enkf_da_all, last_steps=last_steps)


print("Mean RMSE over DA analysis steps in the last 200 filtering steps")
print("----------------------------------------------------------------")
print(f"L63 EnSF    : {mean_l63_ensf:.3f}")
print(f"L63 EnSF-LR : {mean_l63_ensflr:.3f}")
print(f"L63 EnKF    : {mean_l63_enkf:.3f}")
print(f"L96 EnSF    : {mean_l96_ensf:.3f}")
print(f"L96 EnSF-LR : {mean_l96_ensflr:.3f}")
print(f"L96 EnKF    : {mean_l96_enkf:.3f}")


# ============================================================
# Plot combined figure: left Lorenz-63, right Lorenz-96
# ============================================================
fontsize = 12

fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)


# -------------------------
# Panel (a): Lorenz-63
# -------------------------
line1, = axs[0].plot(
    x_l63,
    l63_ensf_mean,
    label="EnSF",
    linewidth=2,
    marker="o",
    markersize=4,
)

line2, = axs[0].plot(
    x_l63,
    l63_ensflr_mean,
    label="EnSF-LR",
    linewidth=2,
    marker="o",
    markersize=4,
)

line3, = axs[0].plot(
    x_l63,
    l63_enkf_mean,
    label="EnKF",
    linewidth=2,
    marker="o",
    markersize=4,
)

axs[0].fill_between(
    x_l63,
    l63_ensf_low,
    l63_ensf_high,
    alpha=alpha_band-0.1,
)

axs[0].fill_between(
    x_l63,
    l63_ensflr_low,
    l63_ensflr_high,
    alpha=alpha_band+0.05,
)

axs[0].fill_between(
    x_l63,
    l63_enkf_low,
    l63_enkf_high,
    alpha=alpha_band+0.05,
)

axs[0].set_title("(a) Lorenz-63", fontsize=fontsize + 2)
axs[0].set_xlabel("Filtering step", fontsize=fontsize + 2)
axs[0].set_ylabel("Full-state RMSE", fontsize=fontsize + 2)
axs[0].set_yticks(np.arange(0, 8.1, 2.0))
axs[0].tick_params(axis="x", labelsize=fontsize + 1)
axs[0].tick_params(axis="y", labelsize=fontsize + 1)
axs[0].grid(True, alpha=0.3)


# -------------------------
# Panel (b): Lorenz-96
# -------------------------
axs[1].plot(
    x_l96,
    l96_ensf_mean,
    label="EnSF",
    linewidth=2,
    marker="o",
    markersize=4,
)

axs[1].plot(
    x_l96,
    l96_ensflr_mean,
    label="EnSF-LR",
    linewidth=2,
    marker="o",
    markersize=4,
)

axs[1].plot(
    x_l96,
    l96_enkf_mean,
    label="EnKF",
    linewidth=2,
    marker="o",
    markersize=4,
)

axs[1].fill_between(
    x_l96,
    l96_ensf_low,
    l96_ensf_high,
    alpha=alpha_band-0.1,
)

axs[1].fill_between(
    x_l96,
    l96_ensflr_low,
    l96_ensflr_high,
    alpha=alpha_band+0.05,
)

axs[1].fill_between(
    x_l96,
    l96_enkf_low,
    l96_enkf_high,
    alpha=alpha_band+0.05,
)

axs[1].set_title("(b) Lorenz-96", fontsize=fontsize + 2)
axs[1].set_xlabel("Filtering step", fontsize=fontsize + 2)
axs[1].set_yticks(np.arange(0, 3.9, 1.0))
axs[1].tick_params(axis="x", labelsize=fontsize + 1)
axs[1].tick_params(axis="y", labelsize=fontsize + 1)
axs[1].grid(True, alpha=0.3)


# ============================================================
# Shared legend at bottom
# ============================================================
plt.tight_layout(rect=[0, 0.18, 1, 1])

fig.legend(
    handles=[line1, line2, line3],
    labels=[
        f"EnSF (L63={mean_l63_ensf:.3f}, L96={mean_l96_ensf:.3f})",
        f"EnSF-LR (L63={mean_l63_ensflr:.3f}, L96={mean_l96_ensflr:.3f})",
        f"EnKF (L63={mean_l63_enkf:.3f}, L96={mean_l96_enkf:.3f})",
    ],
    title="Mean RMSE over DA steps in last 200 filtering steps",
    loc="lower center",
    bbox_to_anchor=(0.52, 0.05),
    ncol=3,
    fontsize=fontsize + 1,
    title_fontsize=fontsize + 1,
    frameon=True,
)


# ============================================================
# Save PNG only
# ============================================================
plt.savefig(
    "figure_linear_L63_L96_combined_rmse_analysis_only_with_ci_seed_avg.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()