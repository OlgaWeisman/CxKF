import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon, Ellipse, Rectangle
from matplotlib.lines import Line2D


# ============================================================
# Settings
# ============================================================

estimated = np.array([-0.25, 0.0])
target = np.array([0.55, 0.0])

xlim = (-3, 3)
ylim = (-3, 3)

# Background grid
gx = np.linspace(xlim[0], xlim[1], 30)
gy = np.linspace(ylim[0], ylim[1], 30)

Gx, Gy = np.meshgrid(gx, gy)


# ============================================================
# DCP - irregular strongly non-convex region
# ============================================================

def dcp_shape(scale=1.0, center=(0.0, 0.0)):

    # Control points for a non-convex DCP shape
    points = np.array([

        # Upper-left ear
        [-1.75,  0.70],
        [-1.78,  0.95],
        [-1.62,  1.25],
        [-1.35,  1.35],
        [-1.05,  1.25],

        # Upper-left indentation
        [-0.70,  1.30],
        [-0.35,  1.22],

        # Upper boundary
        [ 0.05,  1.32],
        [ 0.45,  1.40],
        [ 0.85,  1.40],
        [ 1.20,  1.25],

        # Right outer boundary
        [ 1.48,  1.00],
        [ 1.68,  0.65],
        [ 1.76,  0.25],
        [ 1.70, -0.20],

        # Lower-right ear
        [ 1.50, -0.70],
        [ 1.35, -1.15],
        [ 1.05, -1.25],
        [ 0.85, -1.20],

        # Right inward notch
        [ 0.80, -0.75],
        [ 0.45, -0.72],
        [ 0.20, -0.90],

        # Deep lower concave part
        [-0.05, -1.15],
        [-0.35, -1.40],
        [-0.70, -1.55],
        [-1.10, -1.58],
        [-1.45, -1.45],
        [-1.70, -1.25],

        # Left lower boundary
        [-1.78, -1.00],
        [-1.74, -0.70],
        [-1.58, -0.35],

        # Inner concave boundary
        [-1.32, -0.05],
        [-1.05,  0.12],
        [-0.72,  0.28],
        [-0.45,  0.48],
        [-0.28,  0.70],

        # Return toward upper-left ear
        [-0.20,  0.90],
        [-0.55,  0.88],
        [-0.90,  0.78],
        [-1.25,  0.70],
        [-1.55,  0.70],
    ])


    # --------------------------------------------------------
    # Catmull-Rom interpolation for smooth closed boundary
    # --------------------------------------------------------

    def catmull_rom(p0, p1, p2, p3, n=20):

        t = np.linspace(
            0,
            1,
            n,
            endpoint=False
        )

        t2 = t ** 2
        t3 = t ** 3

        curve = 0.5 * (
            2 * p1
            + (-p0 + p2) * t[:, None]
            + (
                2 * p0
                - 5 * p1
                + 4 * p2
                - p3
            ) * t2[:, None]
            + (
                -p0
                + 3 * p1
                - 3 * p2
                + p3
            ) * t3[:, None]
        )

        return curve


    smooth_points = []

    N = len(points)

    for i in range(N):

        p0 = points[(i - 1) % N]
        p1 = points[i]
        p2 = points[(i + 1) % N]
        p3 = points[(i + 2) % N]

        segment = catmull_rom(
            p0,
            p1,
            p2,
            p3,
            n=15
        )

        smooth_points.append(segment)


    region = np.vstack(smooth_points)

    # Scale
    region[:, 0] *= scale
    region[:, 1] *= scale

    # Translate
    region[:, 0] += center[0]
    region[:, 1] += center[1]

    return region


# ============================================================
# DQR - convex directional region
# ============================================================

def dqr_shape(
    scale=1.0,
    n_vertices=16,
    center=(0.0, 0.0)
):

    theta = np.linspace(
        0,
        2 * np.pi,
        n_vertices,
        endpoint=False
    )

    r = (
        1.45
        + 0.18 * np.cos(theta - 0.4)
        + 0.10 * np.cos(2 * theta)
    )

    x = (
        scale
        * r
        * np.cos(theta)
        + center[0]
    )

    y = (
        scale
        * 1.10
        * r
        * np.sin(theta)
        + center[1]
    )

    return np.column_stack([
        x,
        y
    ])


# ============================================================
# Common subplot formatting
# ============================================================

def setup_axis(ax, title):

    # --------------------------------------------------------
    # Background grid points
    # --------------------------------------------------------

    ax.scatter(
        Gx.ravel(),
        Gy.ravel(),
        s=4,
        color="lightgray",
        alpha=0.45,
        zorder=0
    )


    # --------------------------------------------------------
    # Estimated state
    # --------------------------------------------------------

    ax.scatter(
        estimated[0],
        estimated[1],
        marker="*",
        s=200,
        color="red",
        edgecolor="black",
        linewidth=1,
        zorder=20
    )


    # --------------------------------------------------------
    # Target state
    # --------------------------------------------------------

    ax.scatter(
        target[0],
        target[1],
        marker="*",
        s=200,
        color="green",
        edgecolor="black",
        linewidth=1,
        zorder=20
    )


    # --------------------------------------------------------
    # Title and labels
    # --------------------------------------------------------

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        r"$y_1$",
        fontsize=13
    )

    ax.set_ylabel(
        r"$y_2$",
        fontsize=13
    )


    # --------------------------------------------------------
    # Axis limits
    # --------------------------------------------------------

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_aspect("equal")


    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        True,
        alpha=0.20,
        linewidth=0.6
    )


# ============================================================
# Create 2 x 2 figure
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(8, 8),
    sharex=True,
    sharey=True
)


# ============================================================
# (a) DCP
# ============================================================

ax = axes[0, 0]

dcp_region = dcp_shape(
    scale=1.0
)

ax.add_patch(
    Polygon(
        dcp_region,
        closed=True,
        facecolor="tab:purple",
        edgecolor="black",
        alpha=0.25,
        linewidth=2
    )
)

setup_axis(
    ax,
    "DCP"
)


# ============================================================
# (b) DQR
# ============================================================

ax = axes[0, 1]

dqr_region = dqr_shape(
    scale=1.25,
    n_vertices=16
)

ax.add_patch(
    Polygon(
        dqr_region,
        closed=True,
        facecolor="tab:orange",
        edgecolor="tab:orange",
        alpha=0.25,
        linewidth=2
    )
)

setup_axis(
    ax,
    r"DQR ($|\mathcal{U}|=128$)"
)


# ============================================================
# (c) Elliptical
# ============================================================

ax = axes[1, 0]

ellipse = Ellipse(
    xy=(0.0, 0.0),
    width=4.4,
    height=2.4,
    angle=32,
    facecolor="tab:blue",
    edgecolor="tab:blue",
    alpha=0.25,
    linewidth=2
)

ax.add_patch(ellipse)

setup_axis(
    ax,
    "Elliptical"
)


# ============================================================
# (d) Naive Rectangular
# ============================================================

ax = axes[1, 1]

rectangle = Rectangle(
    (-1.7, -1.3),
    3.5,
    2.6,
    facecolor="tab:green",
    edgecolor="tab:green",
    alpha=0.25,
    linewidth=2
)

ax.add_patch(rectangle)

setup_axis(
    ax,
    "Naive-Rectangular"
)


# ============================================================
# Shared legend
# ============================================================

legend_elements = [

    Line2D(
        [0],
        [0],
        marker="*",
        linestyle="",
        markerfacecolor="red",
        markeredgecolor="black",
        label="Estimated",
        markersize=14
    ),

    Line2D(
        [0],
        [0],
        marker="*",
        linestyle="",
        markerfacecolor="green",
        markeredgecolor="black",
        label="Target",
        markersize=14
    )
]


fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=2,
    fontsize=11,
    bbox_to_anchor=(0.5, 0.01)
)


# ============================================================
# Formatting
# ============================================================

plt.tight_layout(
    rect=[
        0,
        0.07,
        1,
        1
    ]
)


# ============================================================
# Save figure
# ============================================================

plt.savefig(
    "prediction_regions_observation.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "prediction_regions_observation.png",
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# Show
# ============================================================

plt.show()