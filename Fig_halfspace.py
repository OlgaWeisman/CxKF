import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

mu = np.array([0.0, 0.0])

Sigma = np.array([
    [2.5, 1.3],
    [1.3, 1.2]
])

radius = 1.0

direction_sets = [8, 64]


# ============================================================
# Directional quantile
# ============================================================

def directional_quantile(u):

    return (
        u @ mu
        + radius * np.sqrt(u @ Sigma @ u)
    )


# ============================================================
# Compute polygon from directional constraints
# ============================================================

def get_region_vertices(directions):

    vertices = []

    K = len(directions)

    for i in range(K):

        u1 = directions[i]
        u2 = directions[(i + 1) % K]

        q1 = directional_quantile(u1)
        q2 = directional_quantile(u2)

        A = np.vstack([u1, u2])
        b = np.array([q1, q2])

        vertex = np.linalg.solve(A, b)

        vertices.append(vertex)

    return np.array(vertices)


# ============================================================
# Plot one panel
# ============================================================

def plot_directional_region(ax, K):

    # --------------------------------------------------------
    # Directions
    # --------------------------------------------------------

    theta = np.linspace(
        0,
        2 * np.pi,
        K,
        endpoint=False
    )

    directions = np.column_stack([
        np.cos(theta),
        np.sin(theta)
    ])


    # --------------------------------------------------------
    # Compute DQR polygon
    # --------------------------------------------------------

    vertices = get_region_vertices(directions)

    # close polygon
    vertices_closed = np.vstack([
        vertices,
        vertices[0]
    ])


    # --------------------------------------------------------
    # Draw directional hyperplanes
    # --------------------------------------------------------

    line_length = 6

    for i, u in enumerate(directions):

        q = directional_quantile(u)

        # perpendicular direction
        v = np.array([
            -u[1],
             u[0]
        ])

        # closest point to origin on u^T y = q
        p = q * u

        s = np.linspace(
            -line_length,
            line_length,
            100
        )

        line = (
            p[:, None]
            + v[:, None] * s
        )

        ax.plot(
            line[0],
            line[1],
            color="red",
            alpha=0.35 if K > 10 else 0.55,
            linewidth=0.8,
            zorder=1
        )


        # ----------------------------------------------------
        # u arrows for K=8
        # ----------------------------------------------------

        if K <= 8:

            ax.arrow(
                mu[0],
                mu[1],
                0.85 * u[0],
                0.85 * u[1],
                width=0.012,
                head_width=0.10,
                length_includes_head=True,
                color="black",
                zorder=5
            )

            label_pos = mu + 1.03 * u

            ax.text(
                label_pos[0],
                label_pos[1],
                rf"$u_{{{i+1}}}$",
                fontsize=9,
                ha="center",
                va="center"
            )


    # --------------------------------------------------------
    # Draw DQR region
    # --------------------------------------------------------

    ax.plot(
        vertices_closed[:, 0],
        vertices_closed[:, 1],
        color="blue",
        linewidth=2.5,
        zorder=10
    )


    # --------------------------------------------------------
    # Center
    # --------------------------------------------------------

    ax.scatter(
        mu[0],
        mu[1],
        s=20,
        color="black",
        zorder=15
    )


    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_title(
        rf"$|U|={K}$ directions",
        fontsize=13
    )

    ax.set_xlabel(r"$y_1$")
    ax.set_ylabel(r"$y_2$")

    ax.set_aspect("equal")

    ax.set_xlim(-3.8, 3.8)
    ax.set_ylim(-3.8, 3.8)

    ax.grid(False)


# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 4.5)
)

for ax, K in zip(
    axes,
    direction_sets
):

    plot_directional_region(
        ax,
        K
    )


plt.tight_layout()

plt.savefig(
    "directional_quantiles.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "directional_quantiles.svg",
    bbox_inches="tight"
)

plt.show()