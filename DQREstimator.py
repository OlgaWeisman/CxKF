import numpy as np
import ast
from tqdm import tqdm
import torch
import sys
from types import SimpleNamespace
import math
import matplotlib
import copy
from pathlib import Path
from vanilla_qr_nn import VanillaQRNN
from single_layer_qr import SingleLayerPerceptronQR
from sklearn.cluster import KMeans
from matplotlib import patches
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from CP_test import compute_gaussian_quantiles
from scipy.stats import norm

import numpy as np
import matplotlib.pyplot as plt
import torch

y_grid_size_per_y_dim = {
    2: 1e5,
    3: 1e5,
    4: 2e5
}
z_grid_size_per_z_dim = {
    1: 1e4,
    2: 1e4,
    3: 4e4,
    4: 1e5
}
# folder of current script (or project root if running from there)
project_dir = Path(__file__).resolve().parent

# go one level up → PythonCode
base_dir = project_dir.parent

# now append your target subfolder
target_path = base_dir / "mqr-master" / "mqr-master"

sys.path.append(str(target_path))
# sys.path.append(r"C:\Users\owner\Documents\PythonCode\mqr-master\mqr-master")

from losses import multivariate_qr_loss, predict_y, calc_y_u
from helper import generate_directions, get_grid, get_grid_borders_and_stride
from plot_helper import plot_samples
from utils.q_model_ens import MultivariateQuantileModel
from transformations import ConditionalIdentityTransform
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def plot_all_three_same_ax(
    Y,
    zs_to_color,
    quantile_region_radius,
    ellipse_center=None,
    ellipse_Sigma=None,
    c_alpha=None,
    rect_low=None,
    rect_high=None,
    pred_yi=None,   # directional quantiles [e0, e1, -e0, -e1]
    title=None,
    a_alpha=0.9,
    b_alpha=0.05,
    std_pad_factor=0.05
):

    # ------------------------------
    # axis limits (global max bounds)
    # ------------------------------
    bounds_min_list = []
    bounds_max_list = []
    # Y bounds
    Y_cpu = Y.detach().cpu()
    bounds_min_list.append(Y_cpu.min(dim=0).values[:2])
    bounds_max_list.append(Y_cpu.max(dim=0).values[:2])

    # zs bounds (+ radius if circles)
    z_cpu = zs_to_color.detach().cpu()
    z_mins = z_cpu.min(dim=0).values[:2]
    z_maxs = z_cpu.max(dim=0).values[:2]

    fig, ax = plt.subplots()

    # ------------------------------
    # 1) scatter points
    # ------------------------------
    ax.scatter(
        Y[:, 0].cpu().numpy(),
        Y[:, 1].cpu().numpy(),
        s=20,
        alpha=a_alpha
    )

    # ------------------------------
    # 2) quantile region overlay
    # ------------------------------
    if quantile_region_radius is not None:
        circles = [
            plt.Circle(point, radius=quantile_region_radius, linewidth=0)
            for point in zip(*zs_to_color.split(1, dim=1))
        ]
        c = matplotlib.collections.PatchCollection(
            circles,
            facecolor='red',
            edgecolor='none',
            alpha=0.05
        )
        ax.add_collection(c)

        r = float(quantile_region_radius)
        z_mins = z_mins - r
        z_maxs = z_maxs + r
    else:
        ax.scatter(
            zs_to_color[:, 0].cpu().numpy(),
            zs_to_color[:, 1].cpu().numpy(),
            color='yellow',
            alpha=0.05
        )
    bounds_min_list.append(z_mins)
    bounds_max_list.append(z_maxs)

    # ------------------------------
    # 3) plot q_low / q_high borders (±I case)
    # ------------------------------
    if pred_yi is not None:

        q = pred_yi.mean(dim=0)

        # extract borders
        x0_low  = q[0]
        x1_low  = q[1]
        x0_high = (-q[2])
        x1_high = (-q[3])

        # vertical borders
        ax.axvline(x=x0_low, linestyle="--", linewidth=2, color="blue")
        ax.axvline(x=x0_high, linestyle="--", linewidth=2, color="blue")

        # horizontal borders
        ax.axhline(y=x1_low, linestyle="--", linewidth=2, color="blue")
        ax.axhline(y=x1_high, linestyle="--", linewidth=2, color="blue")

        q_mins = torch.tensor([min(x0_low, x0_high),
                               min(x1_low, x1_high)])
        q_maxs = torch.tensor([max(x0_low, x0_high),
                               max(x1_low, x1_high)])

        bounds_min_list.append(q_mins)
        bounds_max_list.append(q_maxs)

    # ------------------------------
    # 4) ellipse overlay
    # ------------------------------
    if ellipse_center is not None and ellipse_Sigma is not None:
        ell_mins, ell_maxs = _add_ellipse_2d(
            ax, ellipse_center, ellipse_Sigma, c_alpha, linewidth=2
        )
        bounds_min_list.append(ell_mins[:2])
        bounds_max_list.append(ell_maxs[:2])

    # ------------------------------
    # 5) rectangle overlay
    # ------------------------------
    if rect_low is not None and rect_high is not None:
        width = (rect_high[0] - rect_low[0]).item()
        height = (rect_high[1] - rect_low[1]).item()

        rect = patches.Rectangle(
            (rect_low[0].item(), rect_low[1].item()),
            width,
            height,
            fill=False,
            linewidth=2,
            edgecolor="black"
        )
        ax.add_patch(rect)

        rect_low_t = torch.as_tensor(rect_low).detach().cpu()
        rect_high_t = torch.as_tensor(rect_high).detach().cpu()

        rect_mins = torch.min(rect_low_t, rect_high_t)
        rect_maxs = torch.max(rect_low_t, rect_high_t)

        bounds_min_list.append(rect_mins[:2])
        bounds_max_list.append(rect_maxs[:2])

    # ------------------------------
    # axis limits
    # ------------------------------
    # combine all bounds
    global_mins = torch.stack(bounds_min_list).min(dim=0).values
    global_maxs = torch.stack(bounds_max_list).max(dim=0).values

    # padding per dimension
    span = (global_maxs - global_mins).clamp(min=1e-6)
    pad_vec = std_pad_factor * span

    ax.set_xlim(float(global_mins[0] - pad_vec[0]),
                float(global_maxs[0] + pad_vec[0]))
    ax.set_ylim(float(global_mins[1] - pad_vec[1]),
                float(global_maxs[1] + pad_vec[1]))

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel("x[0]")
    ax.set_ylabel("x[1]")

    plt.show()
def mahalanobis_elliptical_region(x, x_hat, Sigma, x_train, alpha,bin_id_per_t, plot_data = None):
    """
    Two-sided Mahalanobis region for time-series:

        q_low^2 <= (x - x_hat)^T Σ^{-1} (x - x_hat) <= q_high^2

    Inputs
    ------
    x      : (I, d, T)       true values
    x_hat  : (I, d, T)       predicted means
    Sigma  : (I, d, d, T)    covariance matrices
    alpha  : float           e.g., 0.1 → central 1-α = 90% region

    Returns
    -------
    inside_mask : (T,)       bool per time step (over all iterations)
    mahal_sq    : (I, T)     Mahalanobis distance squared per iter & time
    hat_q_low   : scalar     q_low = sqrt(chi2_{d, α/2})
    hat_q_high  : scalar     q_high = sqrt(chi2_{d, 1-α/2})
    """

    # Convert to torch tensors
    x     = torch.as_tensor(x)
    x_hat = torch.as_tensor(x_hat, device=x.device, dtype=x.dtype)
    Sigma = torch.as_tensor(Sigma, device=x.device, dtype=x.dtype)

    I, d, T = x.shape

    # --------------------------------------------------------
    # Two-sided chi-square quantiles (SciPy) – dimension d
    # --------------------------------------------------------
    q2_low_float  = chi2.ppf(alpha / 2.0,     df=d)
    q2_high_float = chi2.ppf(1.0 - alpha/2.0, df=d)

    r_low  = np.sqrt(q2_low_float)
    r_high = np.sqrt(q2_high_float)

    hat_q_low  = torch.tensor(r_low,  device=x.device, dtype=x.dtype)
    hat_q_high = torch.tensor(r_high, device=x.device, dtype=x.dtype)
    c_alpha_float = chi2.ppf(1.0 - alpha, df=d)
    c_alpha = torch.tensor(c_alpha_float,
                           device=x.device,
                           dtype=x.dtype)                # torch scalar
    # --------------------------------------------------------
    # Mahalanobis distance squared for each (iteration, time)
    # --------------------------------------------------------
    # Reorder dims to treat (I, T) as batch
    # x_perm: (I, T, d), Sigma_perm: (I, T, d, d)
    x_perm     = x.permute(0, 2, 1)
    x_hat_perm = x_hat.permute(0, 2, 1)
    Sigma_perm = Sigma.permute(0, 3, 1, 2)

    diff = x_perm - x_hat_perm                 # (I, T, d)
    diff_row = diff.unsqueeze(-2)              # (I, T, 1, d)
    diff_col = diff.unsqueeze(-1)              # (I, T, d, 1)

    Sigma_inv = torch.linalg.inv(Sigma_perm)   # (I, T, d, d)

    mahal_sq = (diff_row @ Sigma_inv @ diff_col).squeeze(-1).squeeze(-1)  # (I, T)

    inside_per_iter_time = mahal_sq > c_alpha
    # Aggregate over iterations → per time-step mask (T,)
    inside_mask = inside_per_iter_time.float().mean(dim=0)   # (T,)

    # -------------------------
    # "Covered area" via grid proxy:
    # covered_cells ≈ Volume(ellipsoid) / Volume(cell)
    # -------------------------
    grid_size = y_grid_size_per_y_dim[d]
    m = grid_size ** (1.0 / d)  # cells per dimension (float)

    q = 0.01
    border_min = x.quantile(q, dim=0)
    border_max = x.quantile(1 - q, dim=0)
    # pad_ratio = 0.1  # 10%
    # pad = pad_ratio * (border_max - border_min)

    border_min -= 1
    border_max += 1

    stride = (border_max - border_min) / m   # (d, T)
    cell_vol = stride.prod(dim=0)            # (T,)

    # Exact ellipsoid volume:
    # Vol = V_unit_ball(d) * (c_alpha)^(d/2) * sqrt(det(Sigma))
    # where V_unit_ball(d) = pi^(d/2) / Gamma(d/2 + 1)
    v_unit = (math.pi ** (d / 2.0)) / math.gamma(d / 2.0 + 1.0)  # scalar (python float)

    detSigma = torch.linalg.det(Sigma_perm).clamp_min(1e-12)      # (I, T)
    ellip_vol = (v_unit * (c_alpha ** (d / 2.0))) * torch.sqrt(detSigma)  # (I, T)

    covered_cells = ellip_vol #/ cell_vol.unsqueeze(0)             # (I, T)
    covered_area = covered_cells.mean(dim=0)                      # (T,)

    for t in range(T):
        # loop bins that actually appear at this t
        # (exactly what you asked: range(max(bin_id)+1))
        max_bin_t = int(bin_id_per_t[:, t].max().item())

        for b in range(max_bin_t + 1):
            idx = (bin_id_per_t[:, t] == b)
            if idx.sum() == 0:
                continue

            center = x_hat[idx, :, t].mean(dim=0)  # (d,)
            Sigma_mean = Sigma[idx, :, :, t].mean(dim=0)  # (d,d)

            key = (int(t), int(b))
            plot_data.setdefault(key, {})
            plot_data[key].update({
                "ellipse_center": center.detach().cpu(),
                "ellipse_Sigma": Sigma_mean.detach().cpu(),
                "c_alpha": c_alpha.detach().cpu()
            })
            # if idx.sum() == 0:
            #     continue
            #
            # # points in this bin/time
            # Xt = x[idx, :, t]        # (N_bin, d)
            # Xhat_t = x_hat[idx, :, t]  # (N_bin, d)
            # Sig_t = Sigma[idx, :, :, t] # (N_bin, d, d)
            #
            #
            # # --------- plotting in 2D (first two dims) ----------
            # fig, ax = plt.subplots()
            #
            # ax.scatter(
            #     Xt[:, 0].detach().cpu().numpy(),
            #     Xt[:, 1].detach().cpu().numpy(),
            #     s=20
            # )
            #
            # # overlay ellipse if d == 2 (true ellipse)
            # # if d > 2, we still show scatter of dims (0,1) and annotate area
            # if d == 2:
            #     center = Xhat_t.mean(dim=0)          # (2,)
            #     Sigma_mean = Sig_t.mean(dim=0)       # (2,2)
            #     _add_ellipse_2d(ax, center, Sigma_mean, c_alpha.detach().cpu(),
            #                     linewidth=2)
            # base_name = "elliptical_region"
            # # title includes t, bin, and the "appropriate area" for this t
            # fig_name = f"{base_name}_t{t}_bin{b}"
            # area_t = covered_area[t].item() if torch.is_tensor(covered_area) else float(covered_area[t])
            #
            # ax.set_title(f"{fig_name} | covered_area[t]={area_t:.3g}")
            # ax.set_xlabel("x[0]")
            # ax.set_ylabel("x[1]")
            # # ax.set_aspect("equal", adjustable="box")
            # plt.show()

    return inside_mask, covered_area

def naive_rectangular_region_with_gaussian_quantiles(x, x_hat, Sigma, x_train, alpha, bin_id_per_t, plot_data):
    """
    Naïve Multivariate Quantile Regression using provided per-coordinate Gaussian quantiles:

        hat_q_low, hat_q_high = compute_gaussian_quantiles(x_hat, Sigma, alpha)

    Region per coordinate j:
        C^j(x) = [hat_q_low[j], hat_q_high[j]]

    Multivariate region:
        R(x) = C^1 × C^2 × ... × C^d    (axis-aligned rectangle)

    Inputs
    ------
    x      : (I, d, T) true targets
    x_hat  : (I, d, T) predicted means
    Sigma  : (I, d, d, T) covariance matrices
    alpha  : float
    compute_gaussian_quantiles : function returning (hat_q_low, hat_q_high)

    Returns
    -------
    inside_mask : (T,)      fraction of iterations inside rectangle at each t
    inside_iter_time : (I,T) boolean per iteration & time
    hat_q_low   : (I, d, T)
    hat_q_high  : (I, d, T)
    """

    # Convert to tensors
    x     = torch.as_tensor(x)
    x_hat = torch.as_tensor(x_hat, device=x.device, dtype=x.dtype)
    Sigma = torch.as_tensor(Sigma, device=x.device, dtype=x.dtype)

    I, d, T = x.shape

    # -------------------------------------------------------
    # Get per-coordinate Gaussian quantiles
    # -------------------------------------------------------
    hat_q_low, hat_q_high,_ = compute_gaussian_quantiles(x_hat, torch.diagonal(Sigma, dim1=1, dim2=2).permute(0, 2, 1), alpha/d)
    # expected shape: (I, d, T) each

    # -------------------------------------------------------
    # Check per-coordinate membership:
    # low <= x <= high
    # -------------------------------------------------------
    hat_q_low = torch.as_tensor(hat_q_low, device=x.device, dtype=x.dtype)
    hat_q_high = torch.as_tensor(hat_q_high, device=x.device, dtype=x.dtype)

    inside_coord = (x < hat_q_low) | (x > hat_q_high)   # (I, d, T)

    # Rectangle region requires ALL coordinates inside
    inside_iter_time = inside_coord.any(dim=1)            # (I, T)

    # -------------------------------------------------------
    # Fraction of iterations that are inside per time t
    # -------------------------------------------------------
    inside_mask = inside_iter_time.float().mean(dim=0)  # (T,)

    grid_size_total = y_grid_size_per_y_dim[d]
    y_grid_size = grid_size_total ** (1 / d)
    # full_y_grid = get_grid(x_train, y_grid_size, x_train.shape[1], pad=0.2)
    q = 0.01
    border_min = x.quantile(q, dim=0)
    border_max = x.quantile(1 - q, dim=0)

    border_min -= 1
    border_max += 1

    stride = (border_max - border_min) / y_grid_size

    # covered_area = torch.ceil((hat_q_high - hat_q_low)/ stride).prod(dim=1)
    covered_area = torch.ceil((hat_q_high - hat_q_low).prod(dim=1))
    for t in range(T):
        max_bin_t = int(bin_id_per_t[:, t].max().item())

        for b in range(max_bin_t + 1):
            idx = (bin_id_per_t[:, t] == b)
            if idx.sum() == 0:
                continue

            low_bin  = hat_q_low[idx, :, t]    # (Nbin, d)
            high_bin = hat_q_high[idx, :, t]   # (Nbin, d)

            low_mean  = low_bin.mean(dim=0)    # (d,)
            high_mean = high_bin.mean(dim=0)   # (d,)

            # only meaningful as rectangle in 2D, but store anyway
            width  = (high_mean[0] - low_mean[0]).detach()
            height = (high_mean[1] - low_mean[1]).detach() if d >= 2 else torch.tensor(0., device=x.device)

            key = (int(t), int(b))
            plot_data.setdefault(key, {})
            plot_data[key].update({
                "rect_low": low_mean.detach().cpu(),
                "rect_high": high_mean.detach().cpu(),
                "rect_width": width.detach().cpu(),
                "rect_height": height.detach().cpu()
            })
            # idx = (bin_id_per_t[:, t] == b)
            # if idx.sum() == 0:
            #     continue
            #
            # Xt = x[idx, :, t]          # (N_bin, d)
            # # We need per-iteration rectangle bounds to overlay.
            # # Your function computes hat_q_low/high internally, but doesn't return them.
            # # So we re-compute them here in the exact same way (vectorized):
            # #   hat_q_low/high : (I, d, T)
            # diag = torch.diagonal(Sigma, dim1=1, dim2=2).permute(0, 2, 1)  # (I,T,d)
            # hat_q_low, hat_q_high, _ = compute_gaussian_quantiles(
            #     x_hat, diag, alpha / d
            # )
            # hat_q_low  = torch.as_tensor(hat_q_low,  device=x.device, dtype=x.dtype)
            # hat_q_high = torch.as_tensor(hat_q_high, device=x.device, dtype=x.dtype)
            #
            # low_bin  = hat_q_low[idx, :, t]   # (N_bin, d)
            # high_bin = hat_q_high[idx, :, t]  # (N_bin, d)
            #
            # # # optional downsample (still no loops over samples)
            # # if max_points is not None and Xt.shape[0] > max_points:
            # #     perm = torch.randperm(Xt.shape[0], device=x.device)[:max_points]
            # #     Xt = Xt[perm]
            # #     low_bin = low_bin[perm]
            # #     high_bin = high_bin[perm]
            #
            # fig, ax = plt.subplots()
            #
            # # scatter points (show first 2 dims)
            # ax.scatter(
            #     Xt[:, 0].detach().cpu().numpy(),
            #     Xt[:, 1].detach().cpu().numpy(),
            #     s=20
            # )
            #
            # # overlay a *representative* rectangle for this bin/time:
            # # use mean low/high across points in bin (one rectangle)
            # if d == 2:
            #     low_mean  = low_bin.mean(dim=0)   # (2,)
            #     high_mean = high_bin.mean(dim=0)  # (2,)
            #     width  = (high_mean[0] - low_mean[0]).clamp_min(0).item()
            #     height = (high_mean[1] - low_mean[1]).clamp_min(0).item()
            #
            #     rect = patches.Rectangle(
            #         (low_mean[0].item(), low_mean[1].item()),
            #         width,
            #         height,
            #         fill=False,
            #         linewidth=2
            #     )
            #     ax.add_patch(rect)
            #
            # # "appropriate area"
            # # your covered_area is (I,T); show mean area for this bin/time
            # area_bin_t = covered_area[idx, t].float().mean().item()
            # base_name = "rectangular_region"
            # fig_name = f"{base_name}_t{t}_bin{b}"
            # ax.set_title(f"{fig_name} | mean_covered_area={area_bin_t:.3g}")
            #
            # ax.set_xlabel("x[0]")
            # ax.set_ylabel("x[1]")
            # # ax.set_aspect("equal", adjustable="box")
            # plt.show()

    return inside_mask, covered_area

def rectangular_no_learning(x_hat, alpha, d):
    """
    Input
    -----
    x_hat : (I, 2*d, T)

        first d elements -> mean_x
        last  d elements -> sigma_x

    Returns
    -------
    output : (I, 2*d, T)

        [hat_q_low, -hat_q_high]
    """

    # Mean
    mean_x = x_hat[:, :d]

    # Full covariance
    sigma_full = (
            x_hat[:, d:] ** 2
    ).reshape(-1, d, d)

    # Diagonal variances
    sigma_x = torch.diagonal(
        sigma_full,
        dim1=1,
        dim2=2
    )

    # Gaussian quantiles
    hat_q_low, hat_q_high, _ = compute_gaussian_quantiles(
        mean_x,
        sigma_x,
        alpha / d
    )

    hat_q_low = torch.as_tensor(
        hat_q_low,
        device=mean_x.device,
        dtype=mean_x.dtype
    )

    hat_q_high = torch.as_tensor(
        hat_q_high,
        device=mean_x.device,
        dtype=mean_x.dtype
    )

    # ==========================================
    # Interleave:
    # [q_lo^0, -q_hi^0, q_lo^1, -q_hi^1, ...]
    # ==========================================

    output = torch.empty(
        mean_x.shape[0],
        2 * d,
        device=mean_x.device,
        dtype=mean_x.dtype
    )

    output[:, 0::2] = hat_q_low
    output[:, 1::2] = -hat_q_high

    return output

def _add_ellipse_2d(ax, center_2, Sigma_2x2, c_alpha, **kwargs):
    """
    Draw ellipse for (x-center)^T Sigma^{-1} (x-center) = c_alpha
    For 2D: radii = sqrt(c_alpha * eigvals)
    """
    Sigma_2x2 = torch.as_tensor(Sigma_2x2).detach().cpu()
    center_2  = torch.as_tensor(center_2).detach().cpu()

    # eig decomposition
    evals, evecs = torch.linalg.eigh(Sigma_2x2)  # ascending
    evals = torch.clamp(evals, min=1e-12)

    # major/minor radii
    r = torch.sqrt(c_alpha * evals)  # (2,)
    width  = 2.0 * r[1].item()       # largest
    height = 2.0 * r[0].item()       # smallest

    # rotation angle from eigenvector of largest eigenvalue
    v = evecs[:, 1]  # eigenvector for largest eval
    angle = math.degrees(math.atan2(v[1].item(), v[0].item()))

    ell = Ellipse(
        xy=(center_2[0].item(), center_2[1].item()),
        width=width,
        height=height,
        angle=angle,
        fill=False,
        **kwargs
    )
    ax.add_patch(ell)
    # ---- NEW: compute bounding box of ellipse ----
    # conservative bound (safe and simple)
    max_radius = r.max().item()
    mins = center_2 - max_radius
    maxs = center_2 + max_radius

    return mins, maxs


class DQREstimator:
    def __init__(self,
                 hs_str="[64,64]",
                 num_ep=200,
                 num_u=32,
                 batch_size=256,
                 lr=1e-3,
                 wd=0.0,
                 dropout=0.0,
                 device=None,
                 patience=100,
                 debug_plot=False):
        """
        Wrapper class for training and inference of DQR
        using MultivariateQuantileModel from mqr.

        Parameters
        ----------
        hs_str : str
            Hidden dimensions as string, e.g. "[64,64]".
        num_ep : int
            Number of training epochs.
        num_u : int
            Number of directions during training.
        batch_size : int
            Training batch size.
        lr : float
            Learning rate for MultivariateQuantileModel.
        wd : float
            Weight decay.
        dropout : float
            Dropout rate.
        device : torch.device or str or None
            Device. If None, uses cpu.
        patience : int
            num_wait argument for update_va_loss (early stopping patience).
        debug_plot : bool
            If True, plot train vs validation loss at the end of training.
        """
        self.hs = ast.literal_eval(hs_str)
        self.num_ep = num_ep
        self.num_u = num_u
        self.batch_size = batch_size
        self.lr = lr
        self.wd = wd
        self.dropout = dropout
        self.device = torch.device('cpu') if device is None else torch.device(device)
        self.patience = patience
        self.debug_plot = debug_plot

        self.model_ens = None
        self.train_losses = []
        self.eval_losses = []
        self.tau = None  # will be set in fit

    # -----------------------
    # TRAIN
    # -----------------------
    # -----------------------
    # TRAIN
    # -----------------------
    def fit(self, x_train, y_train, x_val, y_val, tau, log_wandb=True):
        """
        Train VanillaQRNN on (x_train, y_train)
        and validate on (x_val, y_val).

        Parameters
        ----------
        x_train : torch.Tensor, shape (N_train, d_x)
        y_train : torch.Tensor, shape (N_train, d_y)
        x_val : torch.Tensor, shape (N_val, d_x)
        y_val : torch.Tensor, shape (N_val, d_y)
        tau : float

        Returns
        -------
        self
        """
        self.tau = tau
        device = self.device

        x_train = x_train.to(device).float()
        y_train = y_train.to(device).float()
        x_val = x_val.to(device).float()
        y_val = y_val.to(device).float()

        x_dim = x_train.shape[1]
        dim_y = y_train.shape[1]

        # # Fixed directions: [e1, e2, ..., -e1, -e2, ...]
        # Q = torch.eye(dim_y, device=device, dtype=torch.float32)
        # u_list = torch.cat([Q, -Q], dim=0)
        u_list = torch.tensor([
            [1., 0.],  # x1 negative direction  -> high
            [-1., 0.],  # x1 positive direction  -> low
            [0., 1.],  # x2 negative direction  -> high
            [0., -1.],  # x2 positive direction  -> low
        ], device=device)
        # if num_u is different, you can later replace this by random/spherical directions
        self.u_list = u_list
        num_u_actual = u_list.shape[0]

        tau_list = torch.full((num_u_actual,), float(tau), device=device)
        # ## Vanilla NN
        # prev_weights = None
        # if hasattr(self, "model"):
        #     prev_weights = copy.deepcopy(self.model.state_dict())
        #
        # self.model = VanillaQRNN(
        #     input_size=x_dim,
        #     num_u=num_u_actual,
        #     hidden_dimensions=self.hs,
        #     dropout=self.dropout
        # ).to(device)
        # if prev_weights is not None:
        #     self.model.load_state_dict(prev_weights)
        # Initilize the weights
        alpha = tau
        hat_q_low = norm.ppf(alpha/2 , loc=0, scale=1)
        hat_q_high = norm.ppf(1 - alpha/2 , loc=0, scale=1)

        hat_q_low = torch.as_tensor(hat_q_low, device=x_train.device, dtype=x_train.dtype)
        hat_q_high = torch.as_tensor(hat_q_high, device=x_train.device, dtype=x_train.dtype)

        W = torch.tensor([
            [1, 0, hat_q_low, 0, 0, 0],
            [-1, 0, -hat_q_high, 0, 0, 0],
            [0, 1, 0, 0, 0, hat_q_low],
            [0, -1, 0, 0, 0, -hat_q_high],
        ], dtype=torch.float32)
        self.model = SingleLayerPerceptronQR(input_size=x_dim, num_u=4)

        self.state_dict_naive = {
            'linear.weight': W
        }


        self.model.load_state_dict(self.state_dict_naive)

        # self.optimizer = torch.optim.Adam(
        #     self.model.parameters(),
        #     lr=self.lr,
        #     weight_decay=self.wd
        # )
        #
        # if log_wandb:
        #     import wandb
        #     wandb.config.update({
        #         "tau": float(tau),
        #         "num_ep": int(self.num_ep),
        #         "batch_size": int(self.batch_size),
        #         "lr": float(self.lr),
        #         "wd": float(self.wd),
        #         "dropout": float(self.dropout),
        #         "hs": str(self.hs),
        #         "num_u": int(num_u_actual),
        #         "x_dim": int(x_train.shape[1]),
        #         "y_dim": int(y_train.shape[1]),
        #     }, allow_val_change=True)
        #
        # loader = DataLoader(
        #     TensorDataset(x_train, y_train),
        #     shuffle=True,
        #     batch_size=self.batch_size,
        # )
        #
        # self.train_losses = []
        # self.eval_losses = []
        #
        # best_val_loss = float("inf")
        # best_model_state = None
        # patience_counter = 0
        #
        # for ep in tqdm(range(self.num_ep), desc="Training VanillaQRNN"):
        #     self.model.train()
        #     ep_train_loss = []
        #
        #     for xi, yi in loader:
        #         self.optimizer.zero_grad()
        #         # hat_q_low, hat_q_high, _ = compute_gaussian_quantiles(xi[:,:2], xi[:,2:3],tau /2)
        #         # hat_q_low = torch.as_tensor(hat_q_low, device=x_train.device, dtype=x_train.dtype)
        #         # hat_q_high = torch.as_tensor(hat_q_high, device=x_train.device, dtype=x_train.dtype)
        #         # N = hat_q_low.shape[0]
        #         # Q = torch.zeros(N, 4, device=hat_q_low.device)
        #         # Q[:, 0] = hat_q_low[:, 0]  # (1,0):   x1 >= q_low
        #         # Q[:, 1] = hat_q_low[:, 1]  # (0,1):   x2 >= q_low
        #         # Q[:, 2] = -hat_q_high[:, 0]  # (-1,0): -x1 >= -q_high → x1 <= q_high
        #         # Q[:, 3] = -hat_q_high[:, 1]  # (0,-1): -x2 >= -q_high → x2 <= q_high
        #         # loss = self.model.loss_fixing(
        #         #     y=yi,
        #         #     x=xi,
        #         #     u_list=u_list,
        #         #     tau_list=tau_list,
        #         #     q_gauss = Q
        #         # )
        #         # loss = self.model.loss_fixing(
        #         #     y=yi,
        #         #     x=xi,
        #         #     u_list=u_list,
        #         #     tau_list=tau_list,
        #         #     q_gauss = Q
        #         # )
        #         loss = self.model.loss(
        #             y=yi,
        #             x=xi,
        #             u_list=u_list,
        #             tau_list=tau_list,
        #         )
        #
        #         loss.backward()
        #         self.optimizer.step()
        #
        #         ep_train_loss.append(loss.item())
        #
        #     ep_tr_loss = float(np.mean(ep_train_loss))
        #     self.train_losses.append(ep_tr_loss)
        #
        #     self.model.eval()
        #     with torch.no_grad():
        #         # hat_q_low, hat_q_high, _ = compute_gaussian_quantiles(x_val[:, :2], x_val[:, 2:3], tau / 2)
        #         # hat_q_low = torch.as_tensor(hat_q_low, device=x_train.device, dtype=x_train.dtype)
        #         # hat_q_high = torch.as_tensor(hat_q_high, device=x_train.device, dtype=x_train.dtype)
        #         # N = hat_q_low.shape[0]
        #         # Q = torch.zeros(N, 4, device=hat_q_low.device)
        #         # Q[:, 0] = hat_q_low[:, 0]  # (1,0):   x1 >= q_low
        #         # Q[:, 1] = hat_q_low[:, 1]  # (0,1):   x2 >= q_low
        #         # Q[:, 2] = -hat_q_high[:, 0]  # (-1,0): -x1 >= -q_high → x1 <= q_high
        #         # Q[:, 3] = -hat_q_high[:, 1]  # (0,-1): -x2 >= -q_high → x2 <= q_high
        #         # ep_va_loss = self.model.loss_fixing(
        #         #     y=y_val,
        #         #     x=x_val,
        #         #     u_list=u_list,
        #         #     tau_list=tau_list,
        #         #     q_gauss = Q
        #         # ).item()
        #         ep_va_loss = self.model.loss(
        #             y=y_val,
        #             x=x_val,
        #             u_list=u_list,
        #             tau_list=tau_list,
        #         ).item()
        #
        #     self.eval_losses.append(ep_va_loss)
        #
        #     if log_wandb:
        #         wandb.log({
        #             "epoch": ep,
        #             "loss/train": ep_tr_loss,
        #             "loss/val": ep_va_loss,
        #         })
        #
        #     print(
        #         f"Epoch {ep+1}/{self.num_ep} | "
        #         f"train_loss={ep_tr_loss:.4f} | val_loss={ep_va_loss:.4f}"
        #     )
        #
        #     # early stopping
        #     if ep_va_loss < best_val_loss:
        #         best_val_loss = ep_va_loss
        #         best_model_state = copy.deepcopy(self.model.state_dict())
        #         patience_counter = 0
        #     else:
        #         patience_counter += 1
        #
        #     if patience_counter >= self.patience:
        #         print(f"Stopping early at epoch {ep+1}")
        #         break
        #
        # if best_model_state is not None:
        #     self.model.load_state_dict(best_model_state)
        #
        # if self.debug_plot:
        #     self._plot_losses()

        return self

    def fit_nonLinear(self, x_train, y_train, x_val, y_val, tau, log_wandb=True):
        """
        Train VanillaQRNN on (x_train, y_train)
        and validate on (x_val, y_val).

        Parameters
        ----------
        x_train : torch.Tensor, shape (N_train, d_x)
        y_train : torch.Tensor, shape (N_train, d_y)
        x_val : torch.Tensor, shape (N_val, d_x)
        y_val : torch.Tensor, shape (N_val, d_y)
        tau : float

        Returns
        -------
        self
        """
        self.tau = tau
        device = self.device

        x_train = x_train.to(device).float()
        y_train = y_train.to(device).float()
        x_val = x_val.to(device).float()
        y_val = y_val.to(device).float()

        x_dim = x_train.shape[1]
        dim_y = y_train.shape[1]

        # m = 64
        # angles = torch.arange(m) * (2 * torch.pi / m) + torch.pi / (2 * m)
        # u_list = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        m = 128
        angles = torch.arange(m) * (2 * torch.pi / m) + torch.pi / (2 * m)

        u_list = torch.stack([
            torch.cos(angles),
            torch.sin(angles)
        ], dim=1)

        self.u_list = u_list
        num_u_actual = u_list.shape[0]

        tau_list = torch.full((num_u_actual,), float(tau), device=device)
        ## Vanilla NN
        # prev_weights = None
        # if hasattr(self, "model"):
        #     prev_weights = copy.deepcopy(self.model.state_dict())

        self.model = VanillaQRNN(
            input_size=x_dim,
            num_u=num_u_actual,
            hidden_dimensions=self.hs,
            dropout=self.dropout
        ).to(device)
        # if prev_weights is not None:
        #     self.model.load_state_dict(prev_weights)
        # Initilize the weights


        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd
        )

        if log_wandb:
            import wandb
            wandb.config.update({
                "tau": float(tau),
                "num_ep": int(self.num_ep),
                "batch_size": int(self.batch_size),
                "lr": float(self.lr),
                "wd": float(self.wd),
                "dropout": float(self.dropout),
                "hs": str(self.hs),
                "num_u": int(num_u_actual),
                "x_dim": int(x_train.shape[1]),
                "y_dim": int(y_train.shape[1]),
            }, allow_val_change=True)

        loader = DataLoader(
            TensorDataset(x_train, y_train),
            shuffle=True,
            batch_size=self.batch_size,
        )

        self.train_losses = []
        self.eval_losses = []

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        for ep in tqdm(range(self.num_ep), desc="Training VanillaQRNN"):
            self.model.train()
            ep_train_loss = []

            for xi, yi in loader:
                self.optimizer.zero_grad()
                loss = self.model.loss(
                    y=yi,
                    x=xi,
                    u_list=u_list,
                    tau_list=tau_list,
                )

                loss.backward()
                self.optimizer.step()

                ep_train_loss.append(loss.item())

            ep_tr_loss = float(np.mean(ep_train_loss))
            self.train_losses.append(ep_tr_loss)

            self.model.eval()
            with torch.no_grad():
                ep_va_loss = self.model.loss(
                    y=y_val,
                    x=x_val,
                    u_list=u_list,
                    tau_list=tau_list,
                ).item()

            self.eval_losses.append(ep_va_loss)

            if log_wandb:
                wandb.log({
                    "epoch": ep,
                    "loss/train": ep_tr_loss,
                    "loss/val": ep_va_loss,
                })

            print(
                f"Epoch {ep+1}/{self.num_ep} | "
                f"train_loss={ep_tr_loss:.4f} | val_loss={ep_va_loss:.4f}"
            )

            # early stopping
            if ep_va_loss < best_val_loss:
                best_val_loss = ep_va_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Stopping early at epoch {ep+1}")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        if self.debug_plot:
            self._plot_losses()

        return self

    def _plot_losses(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label="Train Loss", linewidth=2)
        plt.plot(self.eval_losses, label="Validation Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.grid(True)
        plt.legend()
        plt.show()
    def fit_gaussian_correction_forall_seq(self, x_train, y_train, x_val, y_val, tau, log_wandb=True,load_path = None):
        self.tau = tau
        device = self.device

        x_train = x_train.to(device).float()
        y_train = y_train.to(device).float()
        x_val = x_val.to(device).float()
        y_val = y_val.to(device).float()

        N_train, x_dim, T = x_train.shape
        _, dim_y, T_y = y_train.shape

        m = self.num_u
        u_list = torch.randn(m, dim_y, device=device)

        u_list = u_list / torch.norm(
            u_list,
            dim=1,
            keepdim=True
        )


        self.u_list = u_list
        num_u_actual = u_list.shape[0]
        tau_list = torch.full((num_u_actual,), float(tau), device=device)

        self.model = VanillaQRNN(
            input_size=x_dim,
            num_u=num_u_actual,
            hidden_dimensions=self.hs,
            dropout=self.dropout
        ).to(device)
        if load_path is not None:
            checkpoint = torch.load(
                load_path
            )

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            self.u_list = checkpoint["u_list"]
            self.model.eval()

            print(f"Loaded model from {load_path}")
            return
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd
        )

        if log_wandb:
            import wandb
            wandb.config.update({
                "tau": float(tau),
                "num_ep": int(self.num_ep),
                "batch_size": int(self.batch_size),
                "lr": float(self.lr),
                "wd": float(self.wd),
                "dropout": float(self.dropout),
                "hs": str(self.hs),
                "num_u": int(num_u_actual),
                "x_dim": int(x_dim),
                "y_dim": int(dim_y),
                "T": int(T),
            }, allow_val_change=True)

        loader = DataLoader(
            TensorDataset(x_train, y_train),
            shuffle=True,
            batch_size=self.batch_size,
        )

        self.train_losses = []
        self.eval_losses = []

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        for ep in tqdm(range(self.num_ep), desc="Training VanillaQRNN"):
            self.model.train()
            ep_train_loss = []

            for x_batch, y_batch in loader:
                x_sigma = x_batch[:,dim_y:,:].reshape(x_batch.shape[0], dim_y, dim_y, x_batch.shape[2])**2
                diag = torch.diagonal(x_sigma, dim1=1, dim2=2).permute(0, 2, 1)
                hat_q_low, hat_q_high, _ = compute_gaussian_quantiles(x_batch[:,:dim_y], diag,tau /2)
                hat_q_low = torch.as_tensor(hat_q_low, device=x_train.device, dtype=x_train.dtype)
                self.optimizer.zero_grad()

                total_loss = 0.0
                per_t_train_losses = []

                for t in range(T):
                    x_t = x_batch[:, :, t]  # (B, d_x)
                    y_t = y_batch[:, :, t]  # (B, d_y)

                    loss_t = self.model.loss_fixing(
                        y=y_t,
                        x=x_t,
                        u_list=u_list,
                        tau_list=tau_list,
                        q_gauss=hat_q_low[:,:,t]
                    )

                    total_loss = total_loss + loss_t
                    per_t_train_losses.append(loss_t.item())

                loss = total_loss / T
                loss.backward()
                self.optimizer.step()

                ep_train_loss.append(loss.item())

            ep_tr_loss = float(np.mean(ep_train_loss))
            self.train_losses.append(ep_tr_loss)

            self.model.eval()
            with torch.no_grad():
                val_losses_t = []
                x_sigma = x_val[:,dim_y:,:].reshape(x_val.shape[0], dim_y, dim_y, x_val.shape[2])**2
                diag = torch.diagonal(x_sigma, dim1=1, dim2=2).permute(0, 2, 1)
                hat_q_low, hat_q_high, _ = compute_gaussian_quantiles(x_val[:,:dim_y], diag,tau /2)
                hat_q_low = torch.as_tensor(hat_q_low, device=x_val.device, dtype=x_val.dtype)
                for t in range(T):
                    x_val_t = x_val[:, :, t]
                    y_val_t = y_val[:, :, t]

                    val_loss_t = self.model.loss_fixing(
                        y=y_val_t,
                        x=x_val_t,
                        u_list=u_list,
                        tau_list=tau_list,
                        q_gauss=hat_q_low[:,:,t]
                    ).item()

                    val_losses_t.append(val_loss_t)

                ep_va_loss = float(np.mean(val_losses_t))

            self.eval_losses.append(ep_va_loss)

            if log_wandb:
                import wandb
                log_dict = {
                    "epoch": ep,
                    "loss/train": ep_tr_loss,
                    "loss/val": ep_va_loss,
                }
                for t in range(T):
                    log_dict[f"loss_t/val_t{t}"] = val_losses_t[t]
                wandb.log(log_dict)

            print(
                f"Epoch {ep + 1}/{self.num_ep} | "
                f"train_loss={ep_tr_loss:.4f} | val_loss={ep_va_loss:.4f}"
            )

            if ep_va_loss < best_val_loss:
                best_val_loss = ep_va_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Stopping early at epoch {ep + 1}")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        # Save model weights
        return self

    def fit_nonLinear_forall_seq(self, x_train, y_train, x_val, y_val, tau, log_wandb=True,load_path=None):

        self.tau = tau
        device = self.device

        x_train = x_train.to(device).float()
        y_train = y_train.to(device).float()
        x_val = x_val.to(device).float()
        y_val = y_val.to(device).float()

        N_train, x_dim, T = x_train.shape
        _, dim_y, T_y  = y_train.shape


        m = self.num_u
        u_list = torch.randn(m, dim_y, device=device)

        u_list = u_list / torch.norm(
            u_list,
            dim=1,
            keepdim=True
        )

        self.u_list = u_list
        num_u_actual = u_list.shape[0]
        tau_list = torch.full((num_u_actual,), float(tau), device=device)

        self.model = VanillaQRNN(
            input_size=x_dim,
            num_u=num_u_actual,
            hidden_dimensions=self.hs,
            dropout=self.dropout
        ).to(device)
        #loading
        if load_path is not None:
            checkpoint = torch.load(load_path)

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            self.u_list = checkpoint["u_list"]
            self.model.eval()

            print(f"Loaded model from {load_path}")
            return

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd
        )


        if log_wandb:
            import wandb
            wandb.config.update({
                "tau": float(tau),
                "num_ep": int(self.num_ep),
                "batch_size": int(self.batch_size),
                "lr": float(self.lr),
                "wd": float(self.wd),
                "dropout": float(self.dropout),
                "hs": str(self.hs),
                "num_u": int(num_u_actual),
                "x_dim": int(x_dim),
                "y_dim": int(dim_y),
                "T": int(T),
            }, allow_val_change=True)

        loader = DataLoader(
            TensorDataset(x_train, y_train),
            shuffle=True,
            batch_size=self.batch_size,
        )

        self.train_losses = []
        self.eval_losses = []

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        for ep in tqdm(range(self.num_ep), desc="Training VanillaQRNN"):
            self.model.train()
            ep_train_loss = []

            for x_batch, y_batch in loader:
                self.optimizer.zero_grad()

                total_loss = 0.0
                per_t_train_losses = []

                for t in range(T):
                    x_t = x_batch[:, :, t]  # (B, d_x)
                    y_t = y_batch[:, :, t]  # (B, d_y)

                    loss_t = self.model.loss(
                        y=y_t,
                        x=x_t,
                        u_list=u_list,
                        tau_list=tau_list,
                    )

                    total_loss = total_loss + loss_t
                    per_t_train_losses.append(loss_t.item())

                loss = total_loss / T
                loss.backward()
                self.optimizer.step()

                ep_train_loss.append(loss.item())

            ep_tr_loss = float(np.mean(ep_train_loss))
            self.train_losses.append(ep_tr_loss)

            self.model.eval()
            with torch.no_grad():
                val_losses_t = []

                for t in range(T):
                    x_val_t = x_val[:, :, t]
                    y_val_t = y_val[:, :, t]

                    val_loss_t = self.model.loss(
                        y=y_val_t,
                        x=x_val_t,
                        u_list=u_list,
                        tau_list=tau_list,
                    ).item()

                    val_losses_t.append(val_loss_t)

                ep_va_loss = float(np.mean(val_losses_t))

            self.eval_losses.append(ep_va_loss)

            if log_wandb:
                import wandb
                log_dict = {
                    "epoch": ep,
                    "loss/train": ep_tr_loss,
                    "loss/val": ep_va_loss,
                }
                for t in range(T):
                    log_dict[f"loss_t/val_t{t}"] = val_losses_t[t]
                wandb.log(log_dict)

            print(
                f"Epoch {ep + 1}/{self.num_ep} | "
                f"train_loss={ep_tr_loss:.4f} | val_loss={ep_va_loss:.4f}"
            )

            if ep_va_loss < best_val_loss:
                best_val_loss = ep_va_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Stopping early at epoch {ep + 1}")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        return self

    def _plot_losses(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label="Train Loss", linewidth=2)
        plt.plot(self.eval_losses, label="Validation Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.grid(True)
        plt.legend()
        plt.show()

    def predict(self, x):
        """
        Predict directional thresholds for x.

        Parameters
        ----------
        x : torch.Tensor, shape [N, x_dim] or [x_dim]

        Returns
        -------
        pred : torch.Tensor, shape [N, num_u]
        """
        self.model.eval()

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        x = x.to(self.device).float()
        # hat_q_low, hat_q_high, _ = compute_gaussian_quantiles(x_val[:, :2], x_val[:, 2:3], tau / 2)
        # hat_q_low = torch.as_tensor(hat_q_low, device=x.device, dtype=x.dtype)
        # hat_q_high = torch.as_tensor(hat_q_high, device=x.device, dtype=x.dtype)
        # N = hat_q_low.shape[0]
        # q_gauss = torch.zeros(N, 4, device=hat_q_low.device)
        # q_gauss[:, 0] = hat_q_low[:, 0]  # (1,0):   x1 >= q_low
        # q_gauss[:, 1] = hat_q_low[:, 1]  # (0,1):   x2 >= q_low
        # q_gauss[:, 2] = -hat_q_high[:, 0]  # (-1,0): -x1 >= -q_high → x1 <= q_high
        # q_gauss[:, 3] = -hat_q_high[:, 1]  # (0,-1): -x2 >= -q_high → x2 <= q_high
        with torch.no_grad():
            pred = self.model(x)

        return pred

    def get_min_distance(self,y, points, ignore_zero_distance=False, y_batch_size=50, points_batch_size=10000):
        min_dists_from_points = []

        for i in range(0, y.shape[0], y_batch_size):
            yi = y[i: min(i + y_batch_size, y.shape[0])]
            yi_min_dists_from_points = []
            for j in range(0, points.shape[0], points_batch_size):
                pts = points[j: min(j + points_batch_size, points.shape[0])]
                dist_from_pts = (yi - pts.unsqueeze(1).repeat(1, yi.shape[0], 1)).norm(dim=-1)
                if ignore_zero_distance:
                    dist_from_pts[dist_from_pts == 0] = np.inf
                min_dist_from_pts = dist_from_pts.min(dim=0)[0]
                yi_min_dists_from_points += [min_dist_from_pts]

            if len(yi_min_dists_from_points) > 0:
                min_dists_from_points += [torch.stack(yi_min_dists_from_points)]

        if len(min_dists_from_points) == 0:
            return torch.Tensor([np.inf]).repeat(len(y)).to(y.device)
        else:
            return torch.cat(min_dists_from_points, dim=1).min(dim=0)[0]

    def is_in_region(self, x, y, verbose=True):
        # Elliptical model
        if hasattr(self.model, "mahalanobis_distance"):

            d = y.shape[1]

            mu = x[:, :d]

            Sigma = x[:, d:].reshape(-1, d, d) ** 2
            self.model.lambda_hat = self.lambda_hat
            results = self.model.is_in_region(
                y=y,
                mu=mu,
                Sigma=Sigma
            )

            pred_all = None

            return results, pred_all

        # Regular DQR model
        else:

            return self._is_in_region(x, y)
    def _is_in_region(self, x, y, verbose=True):
        """
        Check whether each y_i is inside the predicted halfspace intersection.

        Parameters
        ----------
        x : torch.Tensor, shape [N, x_dim]
        y : torch.Tensor, shape [N, y_dim]

        Returns
        -------
        coverages : torch.Tensor, shape [N, 1]
        pred_all : torch.Tensor, shape [N, num_u]
        """
        if hasattr(self.model, "eval"):
            self.model.eval()

        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(y.shape) == 1:
            y = y.unsqueeze(0)

        x = x.to(self.device).float()
        y = y.to(self.device).float()

        if self.u_list is None:
            # Q = torch.eye(y.shape[1], device=self.device, dtype=torch.float32)
            # self.u_list = torch.cat([Q, -Q], dim=0)
            u_list = torch.tensor([
                [-1., 0.],  # x1 negative direction  -> high
                [1., 0.],  # x1 positive direction  -> low
                [0., -1.],  # x2 negative direction  -> high
                [0., 1.],  # x2 positive direction  -> low
            ])

        if x.shape[1] < 120:
            batch_size = 10000
        else:
            batch_size = 5000

        idx_range = range(0, y.shape[0], batch_size)
        # if verbose:
        #     idx_range = tqdm(idx_range)

        results = []
        pred_all = []

        with torch.no_grad():
            for i in idx_range:
                xi = x[i:min(i + batch_size, x.shape[0])]
                yi = y[i:min(i + batch_size, y.shape[0])]

                pred_yi = self.model(xi)  # [batch, num_u]
                pred_yi = pred_yi  - self.lambda_hat # calibration
                Y_ui = calc_y_u(self.u_list.to(xi.device), yi)  # [batch, num_u]

                res_i = (Y_ui >= pred_yi).all(dim=1).unsqueeze(1)

                results.append(res_i.detach())
                pred_all.append(pred_yi.detach())

        results = torch.cat(results, dim=0)
        pred_all = torch.cat(pred_all, dim=0)

        return results, pred_all
    def is_in_region_fixing(self, x, y, verbose=True):
        """
        Check whether each y_i is inside the predicted halfspace intersection.

        Parameters
        ----------
        x : torch.Tensor, shape [N, x_dim]
        y : torch.Tensor, shape [N, y_dim]

        Returns
        -------
        coverages : torch.Tensor, shape [N, 1]
        pred_all : torch.Tensor, shape [N, num_u]
        """
        self.model.eval()

        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(y.shape) == 1:
            y = y.unsqueeze(0)

        x = x.to(self.device).float()
        y = y.to(self.device).float()

        if self.u_list is None:
            # Q = torch.eye(y.shape[1], device=self.device, dtype=torch.float32)
            # self.u_list = torch.cat([Q, -Q], dim=0)
            u_list = torch.tensor([
                [-1., 0.],  # x1 negative direction  -> high
                [1., 0.],  # x1 positive direction  -> low
                [0., -1.],  # x2 negative direction  -> high
                [0., 1.],  # x2 positive direction  -> low
            ])

        if x.shape[1] < 120:
            batch_size = 10000
        else:
            batch_size = 5000

        idx_range = range(0, y.shape[0], batch_size)
        # if verbose:
        #     idx_range = tqdm(idx_range)

        results = []
        pred_all = []

        with torch.no_grad():
            for i in idx_range:
                xi = x[i:min(i + batch_size, x.shape[0])]
                yi = y[i:min(i + batch_size, y.shape[0])]
                hat_q_low, hat_q_high, _ = compute_gaussian_quantiles(xi[:, :2], xi[:, 2:3] ** 2, self.tau / 2)
                hat_q_low = torch.as_tensor(hat_q_low, device=xi.device)
                pred_yi = self.model(xi)  # [batch, num_u]
                Y_ui = calc_y_u(self.u_list.to(xi.device), yi)  # [batch, num_u]
                Q_ui = calc_y_u(self.u_list.to(xi.device, dtype=xi.dtype),  hat_q_low.to(device=xi.device, dtype=xi.dtype))
                pred_yi = Q_ui - pred_yi
                res_i = (Y_ui >= pred_yi).all(dim=1).unsqueeze(1)

                results.append(res_i.detach())
                pred_all.append(pred_yi.detach())

        results = torch.cat(results, dim=0)
        pred_all = torch.cat(pred_all, dim=0)

        return results, pred_all

    def get_distance_from_quantile_region(self,
                                          x,
                                          y,
                                          y_stride,
                                          z_grid,
                                          full_y_grid,
                                          tau):
        device = self.device

        x = x.to(device).float()
        y = y.to(device).float()
        z_grid = z_grid.to(device).float()
        full_y_grid = full_y_grid.to(device).float()

        # z_in_region = z_grid[is_in_region(...)]
        z_in_region_mask, _ = self.is_in_region(
            x.repeat(len(z_grid), 1),
            z_grid
        )

        z_in_region_mask = z_in_region_mask.squeeze(1).bool()
        z_in_region = z_grid[z_in_region_mask]

        # Edge case: empty predicted region
        if len(z_in_region) == 0:
            region_distance_as_outside_point = torch.tensor(float("inf"), device=device)
            region_distance_as_inside_point = torch.tensor(float("inf"), device=device)
            covered_area = torch.tensor(0.0, device=device)
            in_region_threshold = torch.tensor(0.0, device=device)
            return (
                region_distance_as_outside_point,
                region_distance_as_inside_point,
                covered_area,
                in_region_threshold
            )

        # Distance between points inside the region
        in_region_distances = self.get_min_distance(
            z_in_region,
            z_in_region,
            ignore_zero_distance=True,
            y_batch_size=10000,
            points_batch_size=10000
        )

        in_region_threshold = torch.quantile(in_region_distances, q=0.9).item()

        # Distance of all grid points to predicted region
        y_grid_dist_from_region = self.get_min_distance(
            full_y_grid,
            z_in_region,
            y_batch_size=20000,
            points_batch_size=5000
        )

        out_of_region_idx = y_grid_dist_from_region > in_region_threshold
        y_out_region = full_y_grid[out_of_region_idx]
        n_out_of_region_out_of_entire_grid = len(y_out_region)
        rnd_idx = np.random.permutation(len(y_out_region))[:1000]
        y_out_region = y_out_region[rnd_idx]


        if len(y_out_region) > 0:
            rnd_idx = np.random.permutation(len(y_out_region))[:20000]
            rnd_idx = torch.as_tensor(rnd_idx, device=device, dtype=torch.long)
            y_out_region = y_out_region[rnd_idx]

            region_distance_as_inside_point = self.get_min_distance(
                y,
                y_out_region
            ).squeeze()
        else:
            region_distance_as_inside_point = torch.tensor(float("inf"), device=device)

        region_distance_as_outside_point = self.get_min_distance(
            y,
            z_in_region
        ).squeeze()

        if self.radius is None:
            covered_area = torch.tensor(
                len(full_y_grid) - n_out_of_region_out_of_entire_grid,
                device=device,
                dtype=torch.float32
            )

        elif self.radius > 0:
            covered_area = (y_grid_dist_from_region < self.radius).float().sum()

        else:
            covered_area = (
                    self.get_min_distance(
                        full_y_grid,
                        y_out_region,
                        y_batch_size=20000,
                        points_batch_size=5000
                    ) > abs(self.radius)
            ).float().sum()

        return (
            region_distance_as_outside_point,
            region_distance_as_inside_point,
            covered_area,
            torch.tensor(in_region_threshold, device=device, dtype=torch.float32)
        )

    def get_distance_from_quantile_region_new(self,
                                          x,
                                          y,
                                          z_grid,
                                          stride):

        device = self.device

        x = x.to(device).float()
        y = y.to(device).float()
        z_grid = z_grid.to(device).float()

        z_grid_width = (z_grid[:, 0].max() - z_grid[:, 0].min())#/stride[0]
        z_grid_height = (z_grid[:, 1].max() - z_grid[:, 1].min())#/stride[1]

        z_in_region_mask, _ = self.is_in_region(
            x.repeat(len(z_grid), 1),
            z_grid
        )
        y_in_region_mask, _ = self.is_in_region(
            x,
            y
        )

        z_in_region_mask = z_in_region_mask.squeeze(1).bool()
        z_in_region = z_grid[z_in_region_mask]

        covered_area = torch.ceil(z_grid_width*z_grid_height*(len(z_in_region)/len(z_grid)))

        # if plot_flag:
        #
        #     # ---------- Convert y to numpy ----------
        #     if isinstance(y, torch.Tensor):
        #         y_plot = y.detach().cpu().numpy().squeeze()
        #     else:
        #         y_plot = np.asarray(y).squeeze()
        #
        #     # ---------- Plot ----------
        #     plt.figure(figsize=(8, 8))
        #
        #     # All grid points
        #     plt.scatter(
        #         z_grid[:, 0],
        #         z_grid[:, 1],
        #         s=5,
        #         alpha=0.2,
        #         label='Grid'
        #     )
        #
        #     # Points inside region
        #     plt.scatter(
        #         z_in_region[:, 0],
        #         z_in_region[:, 1],
        #         s=20,
        #         color='green',
        #         label='In Region'
        #     )
        #
        #     # True point
        #     plt.scatter(
        #         y_plot[0],
        #         y_plot[1],
        #         s=250,
        #         color='red',
        #         marker='*',
        #         edgecolors='black',
        #         label='y'
        #     )
        #
        #     plt.xlabel('x')
        #     plt.ylabel('y')
        #     plt.title('Prediction Region')
        #     plt.legend()
        #     plt.axis('equal')
        #     plt.grid(True)
        #
        #     plt.show()
        return (
            covered_area,
            y_in_region_mask
        )
    def calibrate_crc(self, X_cal, y_cal,alpha):
        """
        Choose lambda according to conformal risk control.

        We want empirical risk <= alpha - B / n
        """
        self.B = 1
        n = len(y_cal)
        threshold = alpha - self.B / n

        if threshold < 0:
            raise ValueError("Calibration set is too small for this alpha and B.")

        y_pred = self.model(X_cal)

        # Candidate lambdas are absolute errors
        Y_ui = calc_y_u(self.u_list.to(y_cal.device), y_cal)  # [batch, num_u]

        # lambda_max = y_pred.max()
        #
        # lambda_candidates = torch.linspace(0, lambda_max, 100)
        lambda_candidates, _ = torch.sort(
            (y_pred - Y_ui).max(dim=1).values
        )
        lambda_candidates = torch.linspace(lambda_candidates.min(), lambda_candidates.max(), 1000)
        i=0
        for lam in lambda_candidates:
            lower_cal = y_pred - lam
            risk = (Y_ui < lower_cal).any(dim=1).float().mean()

            if risk <= threshold:
                self.lambda_hat = lam
                return lam
            i = i+1

        # If no lambda satisfies it, use the largest one
        self.lambda_hat = lambda_candidates[-1]
        return self.lambda_hat

    def calibrate_ellip_crc(self,x_hat, y_cal):
        d = y_cal.shape[1]
        mu_cal = x_hat[:, :d]
        # diagonal covariance
        Sigma_cal = (x_hat[:, d:] ** 2).reshape(-1, d, d)
        n = y_cal.shape[0]

        B = 1.0
        threshold = self.model.alpha - B / n

        with torch.no_grad():

            dist = self.model.mahalanobis_distance(
                y_cal,
                mu_cal,
                Sigma_cal
            )

            residuals = dist - self.model.r0
            candidate_lambdas = torch.linspace(residuals.min(), residuals.max(), 1000)

            for lam in candidate_lambdas:

                radius = self.model.r0 + lam

                risk = (
                        dist > radius
                ).float().mean()

                if risk <= threshold:
                    self.lambda_hat = lam

                    return lam

            self.lambda_hat = candidate_lambdas[-1]

            return self.lambda_hat
    def calibrate(self, x_cal, y_cal, y_train, T, tau):

        self.radius = None
        device = self.device

        x = x_cal.to(device).float()
        y_train = y_train.to(device).float()
        grid_step = z_grid_size_per_z_dim[y_train.shape[1]] ** (1 / y_train.shape[1])
        y_grid_size = y_grid_size_per_y_dim[y_train.shape[1]] ** (1 / y_train.shape[1])
        # z_grid = get_grid(
        #     y_train,
        #     grid_step,
        #     y_train.shape[1],
        #     pad=1
        # )
        q = 0.01
        border_min = y_train.quantile(q, dim=0)
        border_max = y_train.quantile(1 - q, dim=0)
        border_max += 1
        border_min -= 1

        stride = (border_max - border_min) / grid_step
        shifts = [torch.arange(
            border_min[i], border_max[i], step=stride[i], dtype=torch.float32, device=device
        ) for i in range(border_max.shape[0])]

        z_grid = torch.cartesian_prod(*shifts)

        border_min = y_train.quantile(q, dim=0)
        border_max = y_train.quantile(1 - q, dim=0)
        border_max += 0.2
        border_min -= 0.2

        y_stride = (border_max - border_min) / y_grid_size
        shifts = [torch.arange(
            border_min[i], border_max[i], step=y_stride[i], dtype=torch.float32, device=device
        ) for i in range(border_max.shape[0])]

        full_y_grid = torch.cartesian_prod(*shifts)

        y_stride = y_stride.norm()

        z_grid = z_grid.to(device).float()
        full_y_grid = full_y_grid.to(device).float()

        n = len(x)

        region_distance_as_outside_point = torch.zeros(n, device=device)
        region_distance_as_inside_point = torch.zeros(n, device=device)
        covered_area = torch.zeros(n, device=device)
        in_region_threshold = torch.zeros(n, device=device)

        for i in tqdm(range(n)):
            (region_distance_as_outside_point[i],
             region_distance_as_inside_point[i],
             covered_area[i],
             in_region_threshold[i]) = self.get_distance_from_quantile_region(
                x[i].unsqueeze(0),
                y_cal[i].unsqueeze(0),
                y_stride,
                z_grid,
                full_y_grid,
                tau
            )

        is_in_qr = (region_distance_as_outside_point < in_region_threshold)  # is in quantile region
        print(f"calibration coverage before calibration: {np.round(is_in_qr.float().mean().item() * 100, 3)}%")
        # Calibration
        n = len(y_cal)
        q = np.ceil((n + 1) * (1 - tau)) / n
        if is_in_qr.float().mean().item() <= 1 - tau:  # we need to increase the quantile region radius
            scores = region_distance_as_outside_point
            # scores_clean = scores[~torch.isinf(scores)]
            self.radius = torch.quantile(scores, q=q).item()


        else:
            scores = region_distance_as_inside_point
            # scores_clean = scores[~torch.isinf(scores)]
            # self.radius = torch.quantile(-scores_clean, q=q).item()
            self.radius = torch.quantile(-scores, q=q).item()

        I_and_T =  is_in_qr.shape[0]

        if math.isnan(self.radius):
            self.radius = None

        if self.radius is not None:
            self.is_conformalized = True

            if self.radius > 0:
                cal_cov_identifiers = region_distance_as_outside_point < self.radius
            else:
                cal_cov_identifiers = region_distance_as_outside_point > abs(self.radius)

            cal_cov_identifiers = cal_cov_identifiers.reshape(I_and_T // T, T)
            print(
                f"calibration coverage after calibration: {np.round(cal_cov_identifiers.float().mean(dim=0) * 100, 3)}%")
            print("radius: ", np.round(self.radius, 4))


        is_in_qr = is_in_qr.reshape(I_and_T // T, T)
        mean_qr_cov = is_in_qr.float().mean(dim=0)
        total_covered_area = covered_area.reshape(I_and_T // T, T)
        mean_total_covered_area = covered_area.float().mean(dim=0)

        # plot

        print(f"calibration coverage before calibration: {np.round(mean_qr_cov * 100, 3)}%")
        print("total_covered_area:", mean_total_covered_area)
        # return (
        #     region_distance_as_outside_point,
        #     region_distance_as_inside_point,
        #     covered_area,
        #     in_region_threshold
        # )
    def inf(self, x_test, y_test, y_train,  T, tau, radius_flag,t):
        device = self.device
        self.radius = None
        x = x_test.to(device).float()
        y_train = y_train.to(device).float()
        grid_step = z_grid_size_per_z_dim[y_train.shape[1]] ** (1 / y_train.shape[1])
        y_grid_size = y_grid_size_per_y_dim[y_train.shape[1]] ** (1 / y_train.shape[1])

        q = 0.01
        border_min = y_train.quantile(q, dim=0)
        border_max = y_train.quantile(1 - q, dim=0)
        # pad_ratio = 0.5  # 50%
        # pad = pad_ratio * (border_max - border_min)

        border_min -= 1*t
        border_max += 1*t

        stride = (border_max - border_min) / grid_step
        shifts = [torch.arange(
            border_min[i], border_max[i], step=stride[i], dtype=torch.float32, device=device
        ) for i in range(border_max.shape[0])]

        z_grid = torch.cartesian_prod(*shifts)


        border_min = y_train.quantile(q, dim=0)
        border_max = y_train.quantile(1 - q, dim=0)
        border_max += 0.2
        border_min -= 0.2

        y_stride = (border_max - border_min) / y_grid_size
        shifts = [torch.arange(
            border_min[i], border_max[i], step=y_stride[i], dtype=torch.float32, device=device
        ) for i in range(border_max.shape[0])]

        full_y_grid = torch.cartesian_prod(*shifts)

        y_stride = y_stride.norm()

        z_grid = z_grid.to(device).float()
        full_y_grid = full_y_grid.to(device).float()

        n = len(x)

        region_distance_as_outside_point = torch.zeros(n, device=device)
        region_distance_as_inside_point = torch.zeros(n, device=device)
        covered_area = torch.zeros(n, device=device)
        in_region_threshold = torch.zeros(n, device=device)

        for i in tqdm(range(n)):
            (region_distance_as_outside_point[i],
             region_distance_as_inside_point[i],
             covered_area[i],
             in_region_threshold[i]) = self.get_distance_from_quantile_region(
                x[i].unsqueeze(0),
                y_test[i].unsqueeze(0),
                y_stride,
                z_grid,
                full_y_grid,
                tau
            )

        if radius_flag:
            cal_cov_identifiers = region_distance_as_outside_point < self.radius

            I_and_T =  cal_cov_identifiers.shape[0]

            total_covered_area = covered_area.reshape(I_and_T // T, T)
            mean_total_covered_area = total_covered_area.float().mean(dim=0)

            cal_cov_identifiers = cal_cov_identifiers.reshape(I_and_T // T, T)

            mean_cal_cov = cal_cov_identifiers.float().mean(dim=0)
        else:
            is_in_qr = (region_distance_as_outside_point < in_region_threshold)  # is in quantile region
            I_and_T = is_in_qr.shape[0]
            is_in_qr = is_in_qr.reshape(I_and_T // T, T)
            mean_cal_cov = is_in_qr.float().mean(dim=0)

            total_covered_area = covered_area.reshape(I_and_T // T, T)
            mean_total_covered_area = total_covered_area.float().mean(dim=0)

        return 1 - mean_cal_cov, mean_total_covered_area

    def inference_new(self, x_test, y_test, y_train, before_cal_flag=False):
        if before_cal_flag:
            self.lambda_hat = torch.tensor(0.0)
        device = self.device
        x = x_test.to(device).float()
        y_train = y_train.to(device).float()
        grid_step = y_grid_size_per_y_dim[y_train.shape[1]] ** (1 / y_train.shape[1])

        q = 0.01
        border_min = y_train.quantile(q, dim=0)
        border_max = y_train.quantile(1 - q, dim=0)

        border_min -= 2
        border_max += 2

        stride = (border_max - border_min) / grid_step
        shifts = [torch.arange(
            border_min[i], border_max[i], step=stride[i], dtype=torch.float32, device=device
        ) for i in range(border_max.shape[0])]

        z_grid = torch.cartesian_prod(*shifts)

        n = len(x)

        covered_area = torch.zeros(n, device=device)
        in_region = torch.zeros(n, device=device)

        for i in tqdm(range(n)):
            (covered_area[i],
             in_region[i]) = self.get_distance_from_quantile_region_new(
                x[i].unsqueeze(0),
                y_test[i].unsqueeze(0),
                z_grid,
                stride
            )

        error = 1 - in_region.mean()
        mean_total_covered_area = covered_area.mean()
        return error, mean_total_covered_area
    # -----------------------
    # INFERENCE
    # -----------------------
    def inference(self, x_test, untransformed_y_test, untransformed_y_train,transformed_y_train,  T, tau, radius_flag, bin_id, t=None, plot_data=None):
        if radius_flag:
            self.model_ens.radius = self.radius
        else:
            self.model_ens.radius = None


        transform = ConditionalIdentityTransform()
        qr_res = self.model_ens.get_quantile_region_distance(x_test, untransformed_y_test, untransformed_y_train,
                                                   transformed_y_train, transform, tau, get_quantile_region_sample=True)
        region_distance_as_outside_point, region_distance_as_inside_point, total_covered_area, in_region_threshold, pred_yi= \
        qr_res[
            'region_distance_as_outside_point'], \
        qr_res[
            'region_distance_as_inside_point'], \
        qr_res[
            'total_covered_area'], \
        qr_res[
            'in_region_threshold'],\
        qr_res[
            'pred_yi']


        quantile_region_sample = qr_res.get('quantile_region_sample', None)
        quantile_out_region_sample = qr_res.get('quantile_out_region_sample', None)

        if radius_flag:
            cal_cov_identifiers = region_distance_as_outside_point < self.radius

            I_and_T =  cal_cov_identifiers.shape[0]

            total_covered_area = total_covered_area.reshape(I_and_T // T, T)
            mean_total_covered_area = total_covered_area.float().mean(dim=0)

            cal_cov_identifiers = cal_cov_identifiers.reshape(I_and_T // T, T)

            mean_cal_cov = cal_cov_identifiers.float().mean(dim=0)
        else:
            is_in_qr = (region_distance_as_outside_point < in_region_threshold)  # is in quantile region
            I_and_T = is_in_qr.shape[0]
            is_in_qr = is_in_qr.reshape(I_and_T // T, T)
            mean_cal_cov = is_in_qr.float().mean(dim=0)

            total_covered_area = total_covered_area.reshape(I_and_T // T, T)
            mean_total_covered_area = total_covered_area.float().mean(dim=0)

        for b in range(max(bin_id) + 1):
            idx = bin_id == b
            Y = untransformed_y_test[idx]  # (Nbin, 2)

            zs_to_color = torch.cat(
                [quantile_region_sample['y'][i] for i in idx.nonzero()[0]],
                dim=0
            )

            key = (int(t), int(b))
            plot_data.setdefault(key, {})
            plot_data[key].update({
                "t": int(t),
                "b": int(b),
                "Y": Y.detach().cpu(),
                "zs_to_color": zs_to_color.detach().cpu(),
                "quantile_region_radius": self.model_ens.radius,
                "pred_yi": pred_yi
            })

        return 1 - mean_cal_cov, mean_total_covered_area


