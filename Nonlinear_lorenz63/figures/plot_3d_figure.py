import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# --- Load data ---
x_truth = pd.read_csv("x_truth_trajectory_lorenz63_seed4.csv").iloc[:, 1:].to_numpy().copy()
x, y, z = x_truth[:, 0], x_truth[:, 1], x_truth[:, 2]
time = np.arange(len(x_truth))

fig = plt.figure(figsize=(11.5, 8.5))
ax = fig.add_subplot(111, projection="3d")
fontsize = 14

# t_start, t_end, color, label, linewidth, alpha
intervals = [
    (0, 50, "lightgray", "Spin-up", 1.8, 0.55),
    (50, 150, "green", "Quiescent phases", 2.5, 1.0),
    (150, 250, "red", "Transition phase", 2.5, 1.0),
    (250, 300, "green", "_nolegend_", 2.5, 1.0),
]

for t_start, t_end, color, label, linewidth, alpha in intervals:
    idx_start = max(0, t_start)
    idx_end = min(len(time), t_end)

    if idx_end <= idx_start:
        continue

    x_seg = x[idx_start:idx_end]
    y_seg = y[idx_start:idx_end]
    z_seg = z[idx_start:idx_end]

    points = np.array([x_seg, y_seg, z_seg]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = Line3DCollection(
        segments,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        label=label
    )
    ax.add_collection3d(lc)

# Axis limits
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.set_zlim(z.min(), z.max())

# Start and end points
ax.scatter(
    x[0], y[0], z[0],
    color="blue", s=80, label="Start", marker="o",
    edgecolors="black", zorder=5
)

ax.scatter(
    x[-1], y[-1], z[-1],
    color="blue", s=80, label="End", marker="^",
    edgecolors="black", zorder=5
)

# Labels
ax.set_xlabel("$x$", fontsize=fontsize+5, labelpad=10)
ax.set_ylabel("$y$", fontsize=fontsize+5, labelpad=10)
ax.set_zlabel("$z$", fontsize=fontsize+5, labelpad=10)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='z', which='major', labelsize=14)


# Legend
ax.legend(
    loc="center left",
    bbox_to_anchor=(1.1, 0.55),
    fontsize=fontsize - 1,
    framealpha=0.9,
    borderaxespad=0.0
)

# View and grid
ax.view_init(elev=25, azim=-50)
ax.grid(True, linestyle="--", alpha=0.35)
ax.set_box_aspect([1, 1, 1])

plt.tight_layout(rect=[0, 0, 0.82, 1])
plt.savefig("lorenz63_truth_3d_intervals.png", dpi=300, bbox_inches="tight")
plt.show()