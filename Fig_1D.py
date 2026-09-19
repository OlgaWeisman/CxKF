import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

np.random.seed(10)

T = 100
t = np.arange(T)


# ============================================================
# First example
# ============================================================

# True target
true_1 = (
    0.82
    + 0.08 * np.sin(0.18 * t)
    + 0.05 * np.random.randn(T)
)

# Add larger variations around the middle/end
true_1[42:55] += 0.20 * np.random.randn(13) + 0.15
true_1[72:83] += 0.15 * np.random.randn(11) + 0.10
true_1[88:96] += 0.12


# ------------------------------------------------------------
# CQR interval
# ------------------------------------------------------------

cqr_lower_1 = (
    0.64
    + 0.015 * np.sin(0.25 * t)
    + 0.012 * np.random.randn(T)
)

cqr_upper_1 = (
    1.10
    + 0.025 * np.sin(0.08 * t)
    + 0.025 * np.random.randn(T)
)


# ------------------------------------------------------------
# CQKF interval
# ------------------------------------------------------------

cqkf_lower_1 = np.zeros(T)

cqkf_lower_1[:5] = np.linspace(0.58, 0.30, 5)
cqkf_lower_1[5:] = (
    0.21
    + 0.06 * np.sin(0.30 * t[5:])
    + 0.025 * np.random.randn(T - 5)
)

cqkf_upper_1 = (
    1.14
    + 0.025 * np.sin(0.07 * t)
    + 0.018 * np.random.randn(T)
)


# ============================================================
# Second example
# ============================================================

true_2 = (
    0.75 * np.exp(-t / 35)
    + 0.12 * np.sin(0.13 * t)
    + 0.08 * np.random.randn(T)
)


# ------------------------------------------------------------
# CQR interval
# ------------------------------------------------------------

cqr_center_2 = 0.25 * np.exp(-t / 30) - 0.05

cqr_lower_2 = (
    cqr_center_2
    - 0.45
    + 0.06 * np.sin(0.07 * t)
    + 0.025 * np.random.randn(T)
)

cqr_upper_2 = (
    cqr_center_2
    + 0.78
    + 0.05 * np.sin(0.10 * t)
    + 0.025 * np.random.randn(T)
)


# ------------------------------------------------------------
# CQKF interval
# ------------------------------------------------------------

cqkf_center_2 = (
    0.20 * np.exp(-t / 35)
    + 0.05 * np.sin(0.08 * t)
)

cqkf_width_2 = (
    0.48
    + 0.05 * np.sin(0.05 * t)
)

cqkf_lower_2 = (
    cqkf_center_2
    - cqkf_width_2
    + 0.025 * np.random.randn(T)
)

cqkf_upper_2 = (
    cqkf_center_2
    + cqkf_width_2
    + 0.025 * np.random.randn(T)
)


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(15, 9)
)


# ============================================================
# Plot 1
# ============================================================

ax = axes[0]

# CQR region
ax.fill_between(
    t,
    cqr_lower_1,
    cqr_upper_1,
    color="tab:blue",
    alpha=0.35,
    label="CQR"
)

# CQKF region
ax.fill_between(
    t,
    cqkf_lower_1,
    cqkf_upper_1,
    color="tab:orange",
    alpha=0.30,
    label="CGKF"
)

# Region boundaries
ax.plot(
    t,
    cqr_lower_1,
    color="tab:blue",
    linewidth=1.5
)

ax.plot(
    t,
    cqr_upper_1,
    color="tab:red",
    linewidth=1.5
)

ax.plot(
    t,
    cqkf_lower_1,
    color="tab:green",
    linewidth=1.5
)

ax.plot(
    t,
    cqkf_upper_1,
    color="tab:orange",
    linewidth=1.5
)

# True target
ax.plot(
    t,
    true_1,
    "-o",
    color="black",
    linewidth=2,
    markersize=2.5,
    label="True target",
    zorder=10
)

ax.set_title(
    "1D Prediction Intervals: CGKF vs CQKF",
    fontsize=14
)

ax.set_xlabel("Time step")
ax.set_ylabel("Value")

ax.set_xlim(0, T - 1)

ax.grid(
    True,
    alpha=0.25
)

ax.legend(
    loc="upper right"
)


# ============================================================
# Plot 2
# ============================================================

ax = axes[1]

# CQR region
ax.fill_between(
    t,
    cqr_lower_2,
    cqr_upper_2,
    color="tab:blue",
    alpha=0.35,
    label="CQKF"
)

# CQKF region
ax.fill_between(
    t,
    cqkf_lower_2,
    cqkf_upper_2,
    color="tab:orange",
    alpha=0.30,
    label="CQKF"
)

# Region boundaries
ax.plot(
    t,
    cqr_lower_2,
    color="tab:blue",
    linewidth=1.5
)

ax.plot(
    t,
    cqr_upper_2,
    color="tab:red",
    linewidth=1.5
)

ax.plot(
    t,
    cqkf_lower_2,
    color="tab:green",
    linewidth=1.5
)

ax.plot(
    t,
    cqkf_upper_2,
    color="tab:orange",
    linewidth=1.5
)

# True target
ax.plot(
    t,
    true_2,
    "-o",
    color="black",
    linewidth=2,
    markersize=2.5,
    label="True target",
    zorder=10
)

ax.set_title(
    "1D Prediction Intervals: CGKF vs CQKF",
    fontsize=14
)

ax.set_xlabel("Time squnce")
ax.set_ylabel("Value")

ax.set_xlim(0, T - 1)

ax.grid(
    True,
    alpha=0.25
)

ax.legend(
    loc="upper right"
)


# ============================================================
# Layout
# ============================================================

plt.tight_layout()

plt.savefig(
    "CQR_CQKF_prediction_intervals.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "CQR_CQKF_prediction_intervals.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()