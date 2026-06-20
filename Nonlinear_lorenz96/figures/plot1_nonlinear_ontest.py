import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================
OBS_GAP = 10
LAST_STEPS = 200
TCRIT_95_N20 = 2.093

FONT_SIZE = 12
ALPHA_BAND = 0.15

COLOR_ENSF = "C0"
COLOR_ENSFLR = "C1"
COLOR_ENKF = "C2"

OUTPUT_FIGURE = "figure_nonlinear_L63_L96_combined_rmse_analysis_only_seed_avg.png"

L63_SEEDS = list(range(20))
L96_SEEDS = list(range(20, 40))

L63_FILES = {
    "EnSF": "RMSE_lorenz63_nonlinear_hristo_EnSF_obs_x_10obsgap_seed{seed}.csv",
    "EnSF-LR": "RMSE_lorenz63_nonlinear_hristo_EnSF_LR_obs_x_10obsgap_seed{seed}.csv",
    "EnKF": "RMSE_lorenz63_nonlinear_hristo_EnKF_LR_obs_x_10obsgap_seed{seed}.csv",
}

L96_FILES = {
    "EnSF": "RMSE_hristo_EnSF_nonlinear_25%obs_10obsgap_seed{seed}.csv",
    "EnSF-LR": "RMSE_hristo_EnSF_nonlinear_LR_25%obs_10obsgap_seed{seed}.csv",
    "EnKF": "RMSE_hristo_EnKF_nonlinear_25%obs_10obsgap_seed{seed}.csv",
}

METHOD_COLORS = {
    "EnSF": COLOR_ENSF,
    "EnSF-LR": COLOR_ENSFLR,
    "EnKF": COLOR_ENKF,
}

# Make the EnSF band lighter because it often overlaps the others.
METHOD_ALPHAS = {
    "EnSF": ALPHA_BAND - 0.10,
    "EnSF-LR": ALPHA_BAND + 0.05,
    "EnKF": ALPHA_BAND + 0.05,
}


# ============================================================
# Data loading and statistics
# ============================================================
def load_rmse(filename):
    """Load RMSE values from the second column of a CSV file."""
    return pd.read_csv(filename).iloc[:, 1].to_numpy()


def load_all_seed_curves(filename_pattern, seed_list, max_steps=None):
    """
    Load RMSE curves for all seeds.

    Parameters
    ----------
    filename_pattern : str
        Filename pattern containing {seed}.
    seed_list : list
        List of seed indices.
    max_steps : int or None
        If not None, only keep the first max_steps time steps.

    Returns
    -------
    np.ndarray
        Array with shape (n_seeds, n_time_steps).
    """
    curves = []

    for seed in seed_list:
        filename = filename_pattern.format(seed=seed)
        curve = load_rmse(filename)

        if max_steps is not None:
            curve = curve[:max_steps]

        curves.append(curve)

    min_len = min(len(curve) for curve in curves)
    return np.vstack([curve[:min_len] for curve in curves])


def keep_da_steps(curves, obs_gap=OBS_GAP):
    """
    Keep only RMSE values at DA analysis time steps.

    Current convention: analysis values are stored at indices 1, 11, 21, ...
    Change the first value in np.arange from 1 to 0 if your files store
    analysis values at 0, 10, 20, ... instead.
    """
    n_time = curves.shape[1]
    da_idx = np.arange(1, n_time, obs_gap)
    return da_idx, curves[:, da_idx]


def pointwise_mean_ci(curves_da):
    """
    Compute pointwise mean and 95% confidence interval across seeds.

    Parameters
    ----------
    curves_da : np.ndarray
        RMSE values with shape (n_seeds, n_da_times).
    """
    n_seeds = curves_da.shape[0]
    mean = np.mean(curves_da, axis=0)
    std = np.std(curves_da, axis=0, ddof=1)
    half_width = TCRIT_95_N20 * std / np.sqrt(n_seeds)
    return mean, mean - half_width, mean + half_width


def window_mean_ci(x_da, curves_da, last_steps=LAST_STEPS):
    """
    Compute scalar mean RMSE and 95% CI over DA steps in the final window.

    For each seed, the RMSE is first averaged over DA analysis times within
    the last `last_steps` filtering steps. The confidence interval is then
    computed across the seed-level means.
    """
    start_eval = x_da[-1] - last_steps + 1
    eval_mask = x_da >= start_eval

    seed_means = np.mean(curves_da[:, eval_mask], axis=1)
    n_seeds = len(seed_means)

    mean = np.mean(seed_means)
    std = np.std(seed_means, ddof=1)
    half_width = TCRIT_95_N20 * std / np.sqrt(n_seeds)

    return mean, mean - half_width, mean + half_width


def compute_case_statistics(file_patterns, seed_list, max_steps=None):
    """Load one model case and compute all statistics needed for plotting."""
    stats = {}

    for method, pattern in file_patterns.items():
        all_curves = load_all_seed_curves(
            pattern,
            seed_list,
            max_steps=max_steps,
        )

        x_da, curves_da = keep_da_steps(all_curves, obs_gap=OBS_GAP)

        mean_curve, lower_curve, upper_curve = pointwise_mean_ci(curves_da)
        mean_value, lower_value, upper_value = window_mean_ci(
            x_da,
            curves_da,
            last_steps=LAST_STEPS,
        )

        stats[method] = {
            "x_da": x_da,
            "curves_da": curves_da,
            "mean_curve": mean_curve,
            "lower_curve": lower_curve,
            "upper_curve": upper_curve,
            "mean_value": mean_value,
            "ci_lower": lower_value,
            "ci_upper": upper_value,
        }

    return stats


# ============================================================
# Plotting
# ============================================================
def plot_case(ax, stats, title, show_ylabel=False):
    """Plot one panel with mean curves and pointwise confidence bands."""
    line_handles = []

    for method in ["EnSF", "EnSF-LR", "EnKF"]:
        item = stats[method]
        x_da = item["x_da"]
        color = METHOD_COLORS[method]

        ax.fill_between(
            x_da,
            item["lower_curve"],
            item["upper_curve"],
            color=color,
            alpha=METHOD_ALPHAS[method],
            linewidth=0,
        )

        line, = ax.plot(
            x_da,
            item["mean_curve"],
            label=method,
            linewidth=2,
            marker="o",
            markersize=4,
            color=color,
        )
        line_handles.append(line)

    ax.set_title(title, fontsize=FONT_SIZE + 2)
    ax.set_xlabel("Filtering step", fontsize=FONT_SIZE + 2)

    if show_ylabel:
        ax.set_ylabel("Full-state RMSE", fontsize=FONT_SIZE + 2)

    ax.tick_params(axis="x", labelsize=FONT_SIZE + 1)
    ax.tick_params(axis="y", labelsize=FONT_SIZE + 1)
    ax.grid(True, alpha=0.3)

    return line_handles


def print_summary(l63_stats, l96_stats):
    """Print mean RMSE and 95% CI over the final DA-time evaluation window."""
    print("Mean RMSE over DA analysis steps in the last 200 filtering steps")
    print("----------------------------------------------------------------")

    for model_name, stats in [("L63", l63_stats), ("L96", l96_stats)]:
        for method in ["EnSF", "EnSF-LR", "EnKF"]:
            item = stats[method]
            print(
                f"{model_name} {method:<7s}: "
                f"mean = {item['mean_value']:.3f}, "
                f"95% CI = [{item['ci_lower']:.3f}, {item['ci_upper']:.3f}]"
            )


def main():
    l63_stats = compute_case_statistics(L63_FILES,L63_SEEDS,max_steps=None,)
    l96_stats = compute_case_statistics(L96_FILES,L96_SEEDS, max_steps=600,)

    print_summary(l63_stats, l96_stats)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

    handles = plot_case(
        axs[0],
        l63_stats,
        title="(a) Lorenz-63",
        show_ylabel=True,
    )

    plot_case(
        axs[1],
        l96_stats,
        title="(b) Lorenz-96",
        show_ylabel=False,
    )

    plt.tight_layout(rect=[0, 0.18, 1, 1])

    fig.legend(
        handles=handles,
        labels=[
            f"EnSF (L63={l63_stats['EnSF']['mean_value']:.3f}, "
            f"L96={l96_stats['EnSF']['mean_value']:.3f})",
            f"EnSF-LR (L63={l63_stats['EnSF-LR']['mean_value']:.3f}, "
            f"L96={l96_stats['EnSF-LR']['mean_value']:.3f})",
            f"EnKF (L63={l63_stats['EnKF']['mean_value']:.3f}, "
            f"L96={l96_stats['EnKF']['mean_value']:.3f})",
        ],
        title="Mean RMSE over DA steps in last 200 filtering steps",
        loc="lower center",
        bbox_to_anchor=(0.52, 0.05),
        ncol=3,
        fontsize=FONT_SIZE + 1,
        title_fontsize=FONT_SIZE + 1,
        frameon=True,
    )

    plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
