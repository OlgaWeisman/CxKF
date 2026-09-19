import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Data
# ============================================================

x_labels = [
    r'$|\mathcal{U}|=16$',
    r'$|\mathcal{U}|=64$',
    r'$|\mathcal{U}|=128$',
    r'$|\mathcal{U}|=256$'
]

# CQKF
cqkf_mean = np.array([
    66.25,
    53.70,
    53.57,
    53.28
])

cqkf_std = np.array([
    16.28,
    12.35,
    12.34,
    11.54
])

# Elliptical-CGKF
ellip_mean = 51.22
ellip_std = 11.00

# Rec-CGKF
rec_mean = 64.62
rec_std = 13.66

x = np.arange(len(x_labels))

# Colors
cqkf_color = 'tab:blue'
ellip_color = 'tab:orange'
rec_color = 'tab:green'


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 4.2))


# ------------------------------------------------------------
# CQKF
# ------------------------------------------------------------

ax.plot(
    x,
    cqkf_mean,
    'o-',
    color=cqkf_color,
    linewidth=2,
    markersize=7,
    label='CQKF'
)

# CQKF mean ± std
ax.fill_between(
    x,
    cqkf_mean - cqkf_std,
    cqkf_mean + cqkf_std,
    color=cqkf_color,
    alpha=0.20
)


# ------------------------------------------------------------
# Elliptical-CGKF reference
# ------------------------------------------------------------

ax.axhline(
    ellip_mean,
    color=ellip_color,
    linestyle='--',
    linewidth=2,
    label='Elliptical-CGKF'
)

# ax.axhspan(
#     ellip_mean - ellip_std,
#     ellip_mean + ellip_std,
#     color=ellip_color,
#     alpha=0.12
# )


# ------------------------------------------------------------
# Rec-CGKF reference
# ------------------------------------------------------------

ax.axhline(
    rec_mean,
    color=rec_color,
    linestyle='--',
    linewidth=2,
    label='Rec-CGKF'
)

# ax.axhspan(
#     rec_mean - rec_std,
#     rec_mean + rec_std,
#     color=rec_color,
#     alpha=0.12
# )


# ============================================================
# Formatting
# ============================================================

ax.set_xticks(x)
ax.set_xticklabels(x_labels)

ax.set_ylabel('Region Size')
ax.set_xlabel(r'Number of Directions $|\mathcal{U}|$')

ax.grid(axis='y', alpha=0.3)

ax.legend(
    frameon=False,
    loc='upper right'
)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Slight margin around first/last CQKF points
ax.set_xlim(-0.25, len(x_labels) - 0.75)

plt.tight_layout()

plt.savefig(
    'effect_number_of_directions.pdf',
    bbox_inches='tight'
)

plt.savefig(
    'effect_number_of_directions.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()