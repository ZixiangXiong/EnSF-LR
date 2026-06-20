import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read data
y0 = pd.read_csv("RMSE_lorenz63_linear_hristo_EnKF_obs_x_10obsgap_seed0.csv").iloc[:, 1].to_numpy()
y1 = pd.read_csv("RMSE_lorenz63_linear_hristo_EnKF_LR_obs_x_10obsgap_seed0.csv").iloc[:, 1].to_numpy()

y2 = pd.read_csv("RMSE_lorenz96_linear_hristo_EnKF_25%obs_10obsgap_seed0.csv").iloc[:, 1].to_numpy()
y3 = pd.read_csv("RMSE_lorenz96_linear_hristo_EnKF_LR_25%obs_10obsgap_seed0.csv").iloc[:, 1].to_numpy()

# Time/assimilation indices
t63 = np.arange(len(y0))
t96 = np.arange(len(y2))

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

# -------------------------
# Panel (a): Lorenz-63
# -------------------------
axes[0].plot(t63, y0, label="One-step EnKF", linewidth=2)
axes[0].plot(t63, y1, label="Two-step EnKF", linewidth=2, linestyle="--")

axes[0].set_title("(a) Lorenz-63")
axes[0].set_xlabel("Assimilation step")
axes[0].set_ylabel("Full-state RMSE")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# -------------------------
# Panel (b): Lorenz-96
# -------------------------
axes[1].plot(t96, y2, label="One-step EnKF", linewidth=2)
axes[1].plot(t96, y3, label="Two-step EnKF", linewidth=2, linestyle="--")

axes[1].set_title("(b) Lorenz-96")
axes[1].set_xlabel("Assimilation step")
axes[1].set_ylabel("Full-state RMSE")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig("EnKF_one_step_two_step_equivalence_L63_L96.png", dpi=300, bbox_inches="tight")

plt.show()

