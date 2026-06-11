import torch
import math
import numpy as np
import matplotlib
from matplotlib import patches
from matplotlib.patches import Ellipse
from scipy.stats import norm
from scipy.stats import chi2
import matplotlib.pyplot as plt
import sys
import csv
import os
import re
sys.path.append(r"C:\Users\owner\Documents\PythonCode\timeParamCPScores-master\timeParamCPScores-master\code")
from gurobipyTutorial import optimzeTimeAlphasKKTNoMaxLowerBound, optimzeTimeAlphasKKT, optimzeTimeAlphasKKTNoMaxLowerBoundMinArea
from DistSplit import DistSplit
from CP_test import compute_gaussian_quantiles
sys.path.append(r"C:\Users\owner\Documents\PythonCode\mqr-master\mqr-master")
from helper import get_grid_borders_and_stride, get_grid

def mahalanobis_elliptical_region(x, x_hat, Sigma, x_train, alpha,bin_id_per_t):
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
    grid_size = 3e3
    m = grid_size ** (1.0 / d)  # cells per dimension (float)

    pad = 0.2
    q = 0.01
    border_min = x_train.quantile(q, dim=0)        # (d, T)
    border_max = x_train.quantile(1 - q, dim=0)    # (d, T)
    border_max = border_max + pad
    border_min = border_min - pad

    stride = (border_max - border_min) / m   # (d, T)
    cell_vol = stride.prod(dim=0)            # (T,)

    # Exact ellipsoid volume:
    # Vol = V_unit_ball(d) * (c_alpha)^(d/2) * sqrt(det(Sigma))
    # where V_unit_ball(d) = pi^(d/2) / Gamma(d/2 + 1)
    v_unit = (math.pi ** (d / 2.0)) / math.gamma(d / 2.0 + 1.0)  # scalar (python float)

    detSigma = torch.linalg.det(Sigma_perm).clamp_min(1e-12)      # (I, T)
    ellip_vol = (v_unit * (c_alpha ** (d / 2.0))) * torch.sqrt(detSigma)  # (I, T)

    covered_cells = ellip_vol / cell_vol.unsqueeze(0)             # (I, T)
    covered_area = covered_cells.mean(dim=0)                      # (T,)

    for t in range(T):
        # loop bins that actually appear at this t
        # (exactly what you asked: range(max(bin_id)+1))
        max_bin_t = int(bin_id_per_t[:, t].max().item())

        for b in [0]: # range(max_bin_t + 1):
            idx = (bin_id_per_t[:, t] == b)
            if idx.sum() == 0:
                continue

            # points in this bin/time
            Xt = x[idx, :, t]        # (N_bin, d)
            Xhat_t = x_hat[idx, :, t]  # (N_bin, d)
            Sig_t = Sigma[idx, :, :, t] # (N_bin, d, d)

            # if max_points is not None and Xt.shape[0] > max_points:
            #     # quick downsample (no loop)
            #     perm = torch.randperm(Xt.shape[0], device=x.device)[:max_points]
            #     Xt = Xt[perm]
            #     Xhat_t = Xhat_t[perm]
            #     Sig_t = Sig_t[perm]

            # --------- plotting in 2D (first two dims) ----------
            fig, ax = plt.subplots()

            ax.scatter(
                Xt[:, 0].detach().cpu().numpy(),
                Xt[:, 1].detach().cpu().numpy(),
                s=20
            )

            # overlay ellipse if d == 2 (true ellipse)
            # if d > 2, we still show scatter of dims (0,1) and annotate area
            if d == 2:
                center = Xhat_t.mean(dim=0)          # (2,)
                Sigma_mean = Sig_t.mean(dim=0)       # (2,2)
                _add_ellipse_2d(ax, center, Sigma_mean, c_alpha.detach().cpu(),
                                linewidth=2)
            base_name = "elliptical_region"
            # title includes t, bin, and the "appropriate area" for this t
            fig_name = f"{base_name}_t{t}_bin{b}"
            area_t = covered_area[t].item() if torch.is_tensor(covered_area) else float(covered_area[t])

            ax.set_title(f"{fig_name} | covered_area[t]={area_t:.3g}")
            ax.set_xlabel("x[0]")
            ax.set_ylabel("x[1]")
            # ax.set_aspect("equal", adjustable="box")
            plt.show()

    return inside_mask, covered_area

def naive_rectangular_region_with_gaussian_quantiles(x, x_hat, Sigma, x_train, alpha, bin_id_per_t):
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
    # #plot interval
    # title = 'Quantile regression'
    #
    # fig, ax = plt.subplots()  # create once
    #
    # for k in range(I):
    #     y_upper = hat_q_high[k, :, 0].flatten().cpu()
    #     y_lower = hat_q_low[k, :, 0].flatten().cpu()
    #     Y = x[k, :, 0].cpu()
    #
    #     rect = patches.Rectangle(
    #         (y_lower[0], y_lower[1]),
    #         (y_upper - y_lower)[0],
    #         (y_upper - y_lower)[1],
    #         linewidth=1,
    #         edgecolor='r',
    #         facecolor='r',
    #         alpha=0.3  # lower alpha so overlaps are visible
    #     )
    #
    #     ax.add_patch(rect)
    #     ax.scatter(Y[0], Y[1], c='k', s=10)
    #
    # # # axes + cosmetics
    # # ax.set_xlim([0, 1.5])
    # # ax.set_ylim([0, 1.5])
    # ax.set_xlabel("Y0")
    # ax.set_ylabel("Y1")
    # ax.set_title("Naive QR rectangles")
    # plt.show()

    grid_size_total = 3e3
    y_grid_size = grid_size_total ** (1 / d)
    # full_y_grid = get_grid(x_train, y_grid_size, x_train.shape[1], pad=0.2)
    border_max, border_min, stride = get_grid_borders_and_stride(x_train, y_grid_size, pad=0.2)
    covered_area = torch.ceil((hat_q_high - hat_q_low)/ stride).prod(dim=1)
    for t in range(T):
        max_bin_t = int(bin_id_per_t[:, t].max().item())

        for b in [0]: #range(max_bin_t + 1):
            idx = (bin_id_per_t[:, t] == b)
            if idx.sum() == 0:
                continue

            Xt = x[idx, :, t]          # (N_bin, d)
            # We need per-iteration rectangle bounds to overlay.
            # Your function computes hat_q_low/high internally, but doesn't return them.
            # So we re-compute them here in the exact same way (vectorized):
            #   hat_q_low/high : (I, d, T)
            diag = torch.diagonal(Sigma, dim1=1, dim2=2).permute(0, 2, 1)  # (I,T,d)
            hat_q_low, hat_q_high, _ = compute_gaussian_quantiles(
                x_hat, diag, alpha / d
            )
            hat_q_low  = torch.as_tensor(hat_q_low,  device=x.device, dtype=x.dtype)
            hat_q_high = torch.as_tensor(hat_q_high, device=x.device, dtype=x.dtype)

            low_bin  = hat_q_low[idx, :, t]   # (N_bin, d)
            high_bin = hat_q_high[idx, :, t]  # (N_bin, d)

            # # optional downsample (still no loops over samples)
            # if max_points is not None and Xt.shape[0] > max_points:
            #     perm = torch.randperm(Xt.shape[0], device=x.device)[:max_points]
            #     Xt = Xt[perm]
            #     low_bin = low_bin[perm]
            #     high_bin = high_bin[perm]

            fig, ax = plt.subplots()

            # scatter points (show first 2 dims)
            ax.scatter(
                Xt[:, 0].detach().cpu().numpy(),
                Xt[:, 1].detach().cpu().numpy(),
                s=20
            )

            # overlay a *representative* rectangle for this bin/time:
            # use mean low/high across points in bin (one rectangle)
            if d == 2:
                low_mean  = low_bin.mean(dim=0)   # (2,)
                high_mean = high_bin.mean(dim=0)  # (2,)
                width  = (high_mean[0] - low_mean[0]).clamp_min(0).item()
                height = (high_mean[1] - low_mean[1]).clamp_min(0).item()

                rect = patches.Rectangle(
                    (low_mean[0].item(), low_mean[1].item()),
                    width,
                    height,
                    fill=False,
                    linewidth=2
                )
                ax.add_patch(rect)

            # "appropriate area"
            # your covered_area is (I,T); show mean area for this bin/time
            area_bin_t = covered_area[idx, t].float().mean().item()
            base_name = "rectangular_region"
            fig_name = f"{base_name}_t{t}_bin{b}"
            ax.set_title(f"{fig_name} | mean_covered_area={area_bin_t:.3g}")

            ax.set_xlabel("x[0]")
            ax.set_ylabel("x[1]")
            # ax.set_aspect("equal", adjustable="box")
            plt.show()

    return inside_mask, covered_area

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

def metric_calculation(curr_test_target, intervals):
    c = (curr_test_target[:, 0, :].numpy() < intervals[:, :, 0]) | (
                curr_test_target[:, 0, :].numpy() > intervals[:, :, 1])
    s = np.sum(c, -1) > 0
    w = intervals
    return c, s, w

def clalibration_residual_quantiles(train_KF_output_long, train_target_long ,q_low,q_hi, J_T, alpha, title_value, q_half,q_low_aT,q_hi_aT,snr):
    plot_flag = True
    # List of algorithms
    algorithms = ["CQR", "Gaussian only", "CQR_max", "CQKF-Bonf-sw", "Union-TS", "time-series non Union", "Gaussian only long"]#["CQR", "Gaussian only", "CQR_max", "CQR_alpha", "time-series non Union", "CQR_r", "dist-split"]
    #loop on trails
    train_KF_output = train_KF_output_long[:1000, :]
    train_target = train_target_long[:1000, :]
    T = train_KF_output.shape[1]
    # 20% testing 80% calibration
    num_of_calib = int(np.ceil(train_KF_output.size(0) * 0.8))
    num_of_tests = train_KF_output.size(0) - num_of_calib

    num_of_calib_long = int(np.ceil(train_KF_output_long.size(0) * 0.8))
    num_of_tests_long = train_KF_output_long.size(0) - num_of_calib_long

    #Initialization
    c = np.empty([J_T, num_of_tests, T])
    s = np.empty([J_T, num_of_tests])
    w = np.empty([J_T, num_of_tests, T, 2])

    c_long = np.empty([J_T, num_of_tests_long, T])
    s_long = np.empty([J_T, num_of_tests_long])
    w_long = np.empty([J_T, num_of_tests_long, T, 2])

    c_ts = np.empty([J_T, num_of_tests, T])
    s_ts = np.empty([J_T, num_of_tests])
    w_ts = np.empty([J_T, num_of_tests, T, 2])

    c_ts_alpha = np.empty([J_T, num_of_tests, T])
    s_ts_alpha = np.empty([J_T, num_of_tests])
    w_ts_alpha = np.empty([J_T, num_of_tests, T, 2])

    c_ts_only = np.empty([J_T, num_of_tests, T])
    s_ts_only = np.empty([J_T, num_of_tests])
    w_ts_only = np.empty([J_T, num_of_tests, T, 2])

    c_ts_only_union = np.empty([J_T, num_of_tests, T])
    s_ts_only_union = np.empty([J_T, num_of_tests])
    w_ts_only_union = np.empty([J_T, num_of_tests, T, 2])

    c_gu = np.empty([J_T, num_of_tests, T])
    s_gu = np.empty([J_T, num_of_tests])
    w_gu = np.empty([J_T, num_of_tests, T, 2])

    c_gu_long = np.empty([J_T, num_of_tests_long, T])
    s_gu_long = np.empty([J_T, num_of_tests_long])
    w_gu_long = np.empty([J_T, num_of_tests_long, T, 2])

    c_pav = np.empty([J_T, num_of_tests, T])
    s_pav = np.empty([J_T, num_of_tests])
    w_pav = np.empty([J_T, num_of_tests, T, 2])

    c_dist_split = np.empty([J_T, num_of_tests, T])
    s_dist_split = np.empty([J_T, num_of_tests])
    w_dist_split = np.empty([J_T, num_of_tests, T, 2])
    # for non-Union case alpha
    # Estimate alphas
    if "cqr + time-series non Union" in algorithms:
        shuffled_idx_all = torch.randperm(train_KF_output.size(0))
        shuffled_idx_alpha = shuffled_idx_all[:100]
        curr_target_alpha = train_target.index_select(0, shuffled_idx_alpha)
        curr_output_alpha = train_KF_output.index_select(0, shuffled_idx_alpha)
        curr_q_low_alpha = q_low[shuffled_idx_alpha]
        curr_q_hi_alpha = q_hi[shuffled_idx_alpha]

        error_low_a = curr_q_low_alpha - curr_target_alpha.squeeze().numpy()
        error_high_a = curr_target_alpha.squeeze().numpy() - curr_q_hi_alpha
        err_for_alpha = np.abs(np.maximum(error_high_a, error_low_a))
        R_vals = err_for_alpha.tolist()
        m = optimzeTimeAlphasKKTNoMaxLowerBound(R_vals, alpha, 100000)
        alphas = []
        for v in m.getVars():
            if "alphas" in v.varName:
                alphas.append(v.x)
            if "q" in v.varName:
                # print(v.x)
                print("obj: " + str(v.x))
    if "time-series non Union" in algorithms:
        shuffled_idx_all = torch.randperm(train_KF_output.size(0))
        shuffled_idx_alpha = shuffled_idx_all[:100]
        curr_target_alpha = train_target.index_select(0, shuffled_idx_alpha)
        curr_output_alpha = train_KF_output.index_select(0, shuffled_idx_alpha)
        # alphas calibration
        test_target_list = curr_target_alpha.squeeze().tolist()
        x_array_test_list = curr_output_alpha.tolist()
        R_vals = [[math.sqrt((test_target_list[i][j] - x_array_test_list[i][j]) ** 2 ) for j in range(len(test_target_list[i]))] for i in range(len(curr_target_alpha))]

        m = optimzeTimeAlphasKKTNoMaxLowerBound(R_vals, alpha, 100000)
        alphas_v1 = []
        for v in m.getVars():
            if "alphas" in v.varName:
                alphas_v1.append(v.x)
            if "q" in v.varName:
                # print(v.x)
                print("obj: " + str(v.x))


    for j in range(J_T):
        shuffled_idx_long = torch.randperm(train_KF_output_long.size(0))
        curr_output_long = train_KF_output_long.index_select(0, shuffled_idx_long)
        curr_target_long = train_target_long.index_select(0, shuffled_idx_long)
        # for long
        curr_q_low_long = q_low_aT[shuffled_idx_long]
        curr_q_hi_long = q_hi_aT[shuffled_idx_long]

        curr_train_output_long = curr_output_long[:num_of_calib_long]
        curr_train_target_long = curr_target_long[:num_of_calib_long]
        curr_train_q_low_long = curr_q_low_long[:num_of_calib_long]
        curr_train_q_hi_long = curr_q_hi_long[:num_of_calib_long]

        curr_test_output_long = curr_output_long[num_of_calib_long:]
        curr_test_target_long = curr_target_long[num_of_calib_long:]
        curr_test_q_low_long = curr_q_low_long[num_of_calib_long:]
        curr_test_q_hi_long = curr_q_hi_long[num_of_calib_long:]

        # for short
        shuffled_idx = torch.randperm(train_KF_output.size(0))
        curr_output = train_KF_output.index_select(0, shuffled_idx)
        curr_target = train_target.index_select(0, shuffled_idx)
        curr_q_low = q_low[shuffled_idx]
        curr_q_hi = q_hi[shuffled_idx]
        curr_q_half = q_half[shuffled_idx]

        curr_train_output = curr_output[:num_of_calib]
        curr_train_target = curr_target[:num_of_calib]
        curr_train_q_low = curr_q_low[:num_of_calib]
        curr_train_q_hi = curr_q_hi[:num_of_calib]
        curr_train_q_half = curr_q_half[:num_of_calib]

        curr_test_output = curr_output[num_of_calib:]
        curr_test_target = curr_target[num_of_calib:]
        curr_test_q_low = curr_q_low[num_of_calib:]
        curr_test_q_hi = curr_q_hi[num_of_calib:]
        curr_test_q_half = curr_q_half[num_of_calib:]


        # For general calculation
        error_low = curr_train_q_low - curr_train_target.squeeze().numpy()
        error_high = curr_train_target.squeeze().numpy() - curr_train_q_hi
        err = np.maximum(error_high, error_low)
        cal_scores = {0: np.sort(err, 0)[::-1]}
        nc = np.sort(cal_scores[0], 0)

        index = int(np.ceil((1 - alpha) * (nc.shape[0] + 1))) - 1
        index = min(max(index, 0), nc.shape[0] - 1)

        # For long general calculation
        error_low_long = curr_train_q_low_long - curr_train_target_long.squeeze().numpy()
        error_high_long = curr_train_target_long.squeeze().numpy() - curr_train_q_hi_long
        err_long = np.maximum(error_high_long, error_low_long)
        cal_scores_long = {0: np.sort(err_long, 0)[::-1]}
        nc_long = np.sort(cal_scores_long[0], 0)

        index_long = int(np.ceil((1 - alpha/T) * (nc_long.shape[0] + 1))) - 1
        index_long = min(max(index_long, 0), nc_long.shape[0] - 1)
        # CQR
        if "CQR" in algorithms:

            test_err = np.vstack([nc[index, :], nc[index, :]])
            intervals = np.zeros((num_of_tests, T, 2))
            intervals[:, :, 0] = curr_test_q_low - np.tile(test_err[0, :], (num_of_tests, 1))
            intervals[:, :, 1] = curr_test_q_hi + np.tile(test_err[1, :], (num_of_tests, 1))
            # Calculate metric result
            [c[j], s[j], w[j]] = metric_calculation(curr_test_target, intervals)
        if "CQKF-Bonf-sw" in algorithms:
            test_err_long = np.vstack([nc_long[index_long, :], nc_long[index_long, :]])
            intervals_long = np.zeros((num_of_tests_long, T, 2))
            intervals_long[:, :, 0] = curr_test_q_low_long - np.tile(test_err_long[0, :], (num_of_tests_long, 1))
            intervals_long[:, :, 1] = curr_test_q_hi_long + np.tile(test_err_long[1, :], (num_of_tests_long, 1))
            # Calculate metric result
            [c_long[j], s_long[j], w_long[j]] = metric_calculation(curr_test_target_long, intervals_long)

        if "Gaussian only" in algorithms:
            intervals_gu = np.zeros((num_of_tests, T, 2))
            intervals_gu[:, :, 0] = curr_test_q_low
            intervals_gu[:, :, 1] = curr_test_q_hi
            # Calculate metric result
            [c_gu[j], s_gu[j], w_gu[j]] = metric_calculation(curr_test_target, intervals_gu)
        if "Gaussian only long" in algorithms:
            intervals_gu_long = np.zeros((num_of_tests_long, T, 2))
            intervals_gu_long[:, :, 0] = curr_test_q_low_long
            intervals_gu_long[:, :, 1] = curr_test_q_hi_long
            # Calculate metric result
            [c_gu_long[j], s_gu_long[j], w_gu_long[j]] = metric_calculation(curr_test_target_long, intervals_gu_long)

        if "CQR_max" in algorithms:
            # time seq Union Bound
            alpha_err = np.max(err , axis=1)
            cal_scores = {0: np.sort(alpha_err, 0)[::-1]}
            nc = np.sort(cal_scores[0], 0)
            test_err = np.vstack([nc[index], nc[index]])
            intervals_ts = np.zeros((num_of_tests, T, 2))
            intervals_ts[:, :, 0] = curr_test_q_low - np.tile(test_err[0, :], (num_of_tests, T))
            intervals_ts[:, :, 1] = curr_test_q_hi + np.tile(test_err[1, :], (num_of_tests, T))

            # Calculate metric result
            [c_ts[j], s_ts[j], w_ts[j]] = metric_calculation(curr_test_target, intervals_ts)

        # time seq non-Union alpha

        if "CQR_alpha" in algorithms:
            alpha_err = np.max(err * alphas, axis=1)
            cal_scores = {0: np.sort(alpha_err, 0)[::-1]}
            nc = np.sort(cal_scores[0], 0)
            D_cp = [nc[index] / a for a in alphas]
            test_err = np.vstack([D_cp, D_cp])
            intervals_ts_alpha = np.zeros((num_of_tests, T, 2))
            intervals_ts_alpha[:, :, 0] = curr_test_q_low - np.tile(test_err[0, :], (num_of_tests, 1))
            intervals_ts_alpha[:, :, 1] = curr_test_q_hi + np.tile(test_err[1, :], (num_of_tests, 1))
            # Calculate metric result
            [c_ts_alpha[j], s_ts_alpha[j], w_ts_alpha[j]] = metric_calculation(curr_test_target, intervals_ts_alpha)

        if "time-series non Union" in algorithms:
            shuffled_idx_all = torch.randperm(curr_train_target.size(0))
            # shuffled_idx_alpha = shuffled_idx_all[:100]
            # curr_target_alpha = train_target.index_select(0, shuffled_idx_alpha)
            # curr_output_alpha = train_KF_output.index_select(0, shuffled_idx_alpha)
            # # alphas calibration
            # test_target_list = curr_target_alpha.squeeze().tolist()
            # x_array_test_list = curr_output_alpha.tolist()
            # R_vals = [[math.sqrt((test_target_list[i][j] - x_array_test_list[i][j]) ** 2) for j in
            #            range(len(test_target_list[i]))] for i in range(len(curr_target_alpha))]
            #
            # m = optimzeTimeAlphasKKTNoMaxLowerBound(R_vals, alpha, 100000)
            # alphas_v1 = []
            # for v in m.getVars():
            #     if "alphas" in v.varName:
            #         alphas_v1.append(v.x)
            #     if "q" in v.varName:
            #         # print(v.x)
            #         print("obj: " + str(v.x))
            ## Naive cp
            # shuffled_idx_training = shuffled_idx_all[100:]
            curr_train_target_training = train_target[:100, :]#.index_select(0, shuffled_idx_training)
            curr_train_output_training = curr_train_output[:100, :]#.index_select(0, shuffled_idx_training)
            test_target_list = curr_train_target_training.squeeze().tolist()
            x_array_test_list = curr_train_output_training.tolist()
            # calculate region
            R_vals = [
                max([alphas_v1[j] * math.sqrt((test_target_list[i][j] - x_array_test_list[i][j]) ** 2) for j in
                     range(len(test_target_list[i]))]) for i in range(len(test_target_list))]

            R_vals.sort()
            R_vals.append(max(R_vals))
            cal_scores_TS = {0: np.sort(R_vals, 0)[::-1]}
            nc = np.sort(cal_scores_TS[0], 0)

            index_TS = int(np.ceil((1 - alpha) * (nc.shape[0] + 1))) - 1
            index_TS = min(max(index_TS, 0), nc.shape[0] - 1)

            D_cp = R_vals[index_TS]
            D_cp_all = [D_cp / a for a in alphas_v1]
            intervals_ts_only = np.zeros((num_of_tests, T, 2))

            D_cp_all = np.ones(T) * D_cp_all
            intervals_ts_only[:, :, 0] = curr_test_output - np.tile(D_cp_all, (num_of_tests, 1))
            intervals_ts_only[:, :, 1] = curr_test_output + np.tile(D_cp_all, (num_of_tests, 1))

            # Calculate metric result
            [c_ts_only[j], s_ts_only[j], w_ts_only[j]] = metric_calculation(curr_test_target, intervals_ts_only)
        if "Union-TS" in algorithms:
            ## Naive cp
            test_target_list = curr_train_target.squeeze().tolist()
            x_array_test_list = curr_train_output.tolist()
            alphas_v2 = 1#;/T
            # calculate region
            R_vals = [
                max([alphas_v2 * math.sqrt((test_target_list[i][j] - x_array_test_list[i][j]) ** 2) for j in
                     range(len(test_target_list[i]))]) for i in range(len(test_target_list))]

            R_vals.sort()
            R_vals.append(max(R_vals))
            cal_scores_TS = {0: np.sort(R_vals, 0)[::-1]}
            nc_TS = np.sort(cal_scores_TS[0], 0)
            index_TS = int(np.ceil((1 - alpha) * (nc_TS.shape[0] + 1))) - 1
            index_TS = min(max(index_TS, 0), nc_TS.shape[0] - 1)
            D_cp = R_vals[index_TS]

            D_cp_all = D_cp / alphas_v2
            intervals_ts_only_union = np.zeros((num_of_tests, T, 2))

            D_cp_all = np.ones(T) * D_cp_all
            intervals_ts_only_union[:, :, 0] = curr_test_output - np.tile(D_cp_all, (num_of_tests, 1))
            intervals_ts_only_union[:, :, 1] = curr_test_output + np.tile(D_cp_all, (num_of_tests, 1))

            # Calculate metric result
            [c_ts_only_union[j], s_ts_only_union[j], w_ts_only_union[j]] = metric_calculation(curr_test_target, intervals_ts_only_union)


        if "CQR_r" in algorithms:
            error_low = (curr_train_q_low - curr_train_target.squeeze().numpy())/(curr_train_q_half - curr_train_q_low)
            error_high = (curr_train_target.squeeze().numpy() - curr_train_q_hi)/(curr_train_q_hi - curr_train_q_half)
            err = np.maximum(error_high, error_low)
            cal_scores = {0: np.sort(err, 0)[::-1]}
            nc = np.sort(cal_scores[0], 0)
            test_err = np.vstack([nc[index, :], nc[index, :]])
            intervals_cqr_m = np.zeros((num_of_tests, T, 2))
            intervals_cqr_m[:, :, 0] = curr_test_q_low - np.tile(test_err[0, :], (num_of_tests, 1))*(curr_test_q_half -curr_test_q_low)
            intervals_cqr_m[:, :, 1] = curr_test_q_hi + np.tile(test_err[1, :], (num_of_tests, 1))*(curr_test_q_hi -curr_test_q_half)

            # Calculate metric result
            [c_pav[j], s_pav[j], w_pav[j]] = metric_calculation(curr_test_target, intervals_cqr_m)

        if "dist-split" in algorithms:
            # Dist-split
            intervals_dist_split = np.zeros((num_of_tests, T, 2))
            for t in range(T):
                fit = DistSplit.run_prediction_bands_only(curr_train_output[:, t].unsqueeze(-1).numpy(), curr_train_target[:, :, t].squeeze().numpy())
                ths = [curr_test_q_low[:, t], curr_test_q_hi[:, t]]
                results = DistSplit.predict_dist_split_r(fit, curr_test_output[:, t].unsqueeze(-1).numpy(), curr_test_target[:, :, t].squeeze().numpy(), alpha)
                intervals_dist_split[:, t, 0] = results['lower_dist']
                intervals_dist_split[:, t, 1] = results['upper_dist']
            [c_dist_split[j], s_dist_split[j], w_dist_split[j]] = metric_calculation(curr_test_target, intervals_dist_split)

        if plot_flag:
            if j == 0:

                plt.plot(curr_test_target[0, 0, :], "bo")
                # plt.fill_between(
                #     np.arange(T), intervals[0,:, 0], intervals[0, :, 1], alpha=0.2, color="#0072B2",
                #     label="Interval CQKF-sw")

                plt.fill_between(
                    np.arange(T), intervals_ts[0,:, 0], intervals_ts[0, :, 1], alpha=0.2, color="#0072B2",
                    label="Interval CQKF-tw")
                # plt.fill_between(
                #     np.arange(T), intervals_long[0,:, 0], intervals_long[0, :, 1], alpha=0.2, color="#009E73",
                #     label="Interval CQKF-Bonf-sw")
                # plt.fill_between(
                #     np.arange(T), intervals_ts_only[0,:, 0], intervals_ts_only[0, :, 1], alpha=0.2, color="#CC79A7",
                #     label="Interval Residuals-LCP")

                plt.fill_between(
                    np.arange(T), intervals_ts_only_union[0,:, 0], intervals_ts_only_union[0, :, 1], alpha=0.2, color="#E69F00",
                    label="Interval Residuals-tw")
                # plt.fill_between(
                #     np.arange(T), intervals_gu_long[0,:, 0], intervals_ts_only[0, :, 1], alpha=0.2, color="#56B4E9",
                #     label="Pred. interval Gauss-Bonf-sw")
                # plt.fill_between(
                #     np.arange(T), intervals_gu[0, :, 0], intervals_gu[0, :, 1], alpha=0.2, color="#F0E442",
                #     label="Interval Gauss-sw")
                plt.xlabel("Sequence")
                plt.ylabel("Values and prediction intervals")
                plt.legend(loc="best")
                plt.grid()
                # plt.title("CQR")
                plt.show()

    c_pav_mean = np.mean(np.mean(c_pav, 1), 0) * 100
    s_pav_mean = np.mean(s_pav) * 100
    w_pav_mean = np.mean(np.mean(np.diff(w_pav), 1), 0)

    c_ts_alpha_mean = np.mean(np.mean(c_ts_alpha, 1), 0)*100
    s_ts_alpha_mean = np.mean(s_ts_alpha)*100
    w_ts_alpha_mean = np.mean(np.mean(np.diff(w_ts_alpha), 1), 0)


    c_gu_mean = np.mean(np.mean(c_gu, 1), 0)*100
    s_gu_mean = np.mean(s_gu)*100
    w_gu_mean = np.mean(np.mean(np.diff(w_gu), 1), 0)

    c_mean = np.mean(np.mean(c, 1), 0)*100
    s_mean = np.mean(s)*100
    w_mean = np.mean(np.mean(np.diff(w), 1), 0)

    c_mean_long = np.mean(np.mean(c_long, 1), 0)*100
    s_mean_long = np.mean(s_long)*100
    w_mean_long = np.mean(np.mean(np.diff(w_long), 1), 0)

    c_gu_mean_long = np.mean(np.mean(c_gu_long, 1), 0)*100
    s_gu_mean_long = np.mean(s_gu_long)*100
    w_gu_mean_long = np.mean(np.mean(np.diff(w_gu_long), 1), 0)

    c_ts_mean = np.mean(np.mean(c_ts,1), 0)*100
    s_ts_mean = np.mean(s_ts)*100
    w_ts_mean = np.mean(np.mean(np.diff(w_ts),1), 0)
    #
    c_ts_only_mean = np.mean(np.mean(c_ts_only,1), 0)*100
    s_ts_only_mean = np.mean(s_ts_only)*100
    w_ts_only_mean = np.mean(np.mean(np.diff(w_ts_only),1), 0)

    c_ts_only_union_mean = np.mean(np.mean(c_ts_only_union,1), 0)*100
    s_ts_only_union_mean = np.mean(s_ts_only_union)*100
    w_ts_only_union_mean = np.mean(np.mean(np.diff(w_ts_only_union),1), 0)

    c_dist_split_mean = np.mean(np.mean(c_dist_split,1), 0)*100
    s_dist_split_mean = np.mean(s_dist_split)*100
    w_dist_split_mean = np.mean(np.mean(np.diff(w_dist_split),1), 0)
    ## Insert to file
    writer = ScenarioMetricsWriter(scenario="non-Gauss", outdir="outputs")
    writer.append(algo="CQKF-TjW", snr_db=snr,
                  C_mean=c_ts_mean.mean(), C_svd=c_ts_mean.std(), S=s_ts_mean, WI_mean=w_ts_mean.mean(), WI_svd=w_ts_mean.std())
    writer.append(algo="CQKF-sw", snr_db=snr,
                  C_mean=c_mean.mean(), C_svd=c_mean.std(), S=s_mean, WI_mean=w_mean.mean(), WI_svd=w_mean.std())
    writer.append(algo="LCP-TS", snr_db=snr,
                  C_mean=c_ts_only_mean.mean(), C_svd=c_ts_only_mean.std(), S=s_ts_only_mean, WI_mean=w_ts_only_mean.mean(), WI_svd=w_ts_only_mean.std())
    writer.append(algo="CQKF-Bonf-sw", snr_db=snr,
                  C_mean=c_mean_long.mean(), C_svd=c_mean_long.std(), S=s_mean_long, WI_mean=w_mean_long.mean(), WI_svd=w_mean_long.std())
    writer.append(algo="KF-Gauss", snr_db=snr,
                  C_mean=c_gu_mean.mean(), C_svd=c_gu_mean.std(), S=s_gu_mean, WI_mean=w_gu_mean.mean(), WI_svd=w_gu_mean.std())
    writer.append(algo="KF-Gauss-Bonf", snr_db=snr,
                  C_mean=c_gu_mean_long.mean(), C_svd=c_gu_mean_long.std(), S=s_gu_mean_long, WI_mean=w_gu_mean_long.mean(), WI_svd=w_gu_mean_long.std())
    writer.append(algo="Union-TS", snr_db=snr,
                  C_mean=c_ts_only_union_mean.mean(), C_svd=c_ts_only_union_mean.std(), S=s_ts_only_union_mean, WI_mean=w_ts_only_union_mean.mean(), WI_svd=w_ts_only_union_mean.std())
    if "CQR_max" in algorithms:
        mean_pr_cp_ts = s_ts_mean
        plt.plot(c_ts_mean, label=f'CQKF-tw (TrjFail={mean_pr_cp_ts:.3f} %)')
    if 0:#"CQR_alpha" in algorithms:
        mean_pr_cp_ts = s_ts_alpha_mean
        plt.plot(c_ts_alpha_mean, label=f'CQKF-TjW-alpha (TrjFail={mean_pr_cp_ts:.3f} %)')

    if "time-series non Union" in algorithms:
        mean_pr_cp_ts = s_ts_only_mean
        plt.plot(c_ts_only_mean, label=f'Residuals-LCP (TrjFail={mean_pr_cp_ts:.3f} %)')
    if "Union-TS" in algorithms:
        mean_pr_cp_ts = s_ts_only_union_mean
        plt.plot(c_ts_only_union_mean, label=f'Residuals-tw (TrjFail={mean_pr_cp_ts:.3f} %)')

    if "CQR" in algorithms:
        mean_pr_cqr = s_mean
        plt.plot(c_mean, label=f'CQKF-sw (TrjFail={mean_pr_cqr:.3f} %)')
    if "CQKF-Bonf-sw" in algorithms:
        mean_pr_cqr = s_mean_long
        plt.plot(c_mean_long, label=f'CQKF-Bonf-sw (TrjFail={mean_pr_cqr:.3f} %)')
    if "Gaussian only" in algorithms:
        mean_pr_cp_gu = s_gu_mean
        plt.plot(c_gu_mean, label=f'Gauss-sw (TrjFail={mean_pr_cp_gu:.3f} %)')
    if "Gaussian only long" in algorithms:
            mean_pr_cp_gu = s_gu_mean_long
            plt.plot(c_gu_mean_long, label=f'Gauss-Bonf-sw (TrjFail={mean_pr_cp_gu:.3f} %)')
    if 0:#"CQR_r" in algorithms:
        plt.plot(c_pav_mean, label=f'CQKF-sw-r (mean={s_pav_mean:.3f} %)')
    if "dist-split" in algorithms:
        plt.plot(c_dist_split_mean, label=f'dist-split (mean={s_dist_split_mean:.3f} %)')
    plt.xlabel("Sequence")
    plt.ylabel("Error %")
    plt.title(f"1/r2 [dB]: {title_value:.2f}")
    plt.legend()
    plt.grid()
    plt.show()

    if "CQR_max" in algorithms:
        mean_pr_cp_ts = w_ts_mean.mean()
        plt.plot(w_ts_mean, label=f'CQKF-tw (mean width={mean_pr_cp_ts:.3f} )')
    if "CQR_alpha" in algorithms:
        mean_pr_cp_ts = w_ts_alpha_mean.mean()
        plt.plot(w_ts_alpha_mean, label=f'CQKF-TjW-alpha (mean width={mean_pr_cp_ts:.3f} )')
    if "time-series non Union" in algorithms:
        mean_pr_cp_ts = w_ts_only_mean.mean()
        plt.plot(w_ts_only_mean, label=f'Residuals-LCP (mean width={mean_pr_cp_ts:.3f})')
    if "Union-TS" in algorithms:
        mean_pr_cp_ts = w_ts_only_union_mean.mean()
        plt.plot(w_ts_only_union_mean, label=f'Residuals-tw (mean width={mean_pr_cp_ts:.3f})')
    if "CQR" in algorithms:
        mean_pr_cqr = w_mean.mean()
        plt.plot(w_mean, label=f'CQKF-sw (mean width={mean_pr_cqr:.3f})')
    if "CQKF-Bonf-sw" in algorithms:
        mean_pr_cqr = w_mean_long.mean()
        plt.plot(w_mean_long, label=f'CQKF-Bonf-sw (mean width={mean_pr_cqr:.3f})')
    if "Gaussian only" in algorithms:
        mean_pr_cp_gu = w_gu_mean.mean()
        plt.plot(w_gu_mean, label=f'Gauss-sw  (mean width={mean_pr_cp_gu:.3f})')
    if "Gaussian only long" in algorithms:
        mean_pr_cp_gu = w_gu_mean_long.mean()
        plt.plot(w_gu_mean_long, label=f'Gauss-Bonf-sw  (mean width={mean_pr_cp_gu:.3f})')
    if "CQR_r" in algorithms:
        mean_pr_cp_gu = w_pav_mean.mean()
        plt.plot(w_pav_mean, label=f'CQKF-sw-r (mean width={mean_pr_cp_gu:.3f})')
    if "dist-split" in algorithms:
        mean_pr_cp_gu = w_dist_split_mean.mean()
        plt.plot(w_dist_split_mean, label=f'dist-split (mean width={mean_pr_cp_gu:.3f})')
    plt.xlabel("Sequence")
    plt.ylabel("Interval Width")
    plt.title(f"1/r2 [dB]: {title_value:.2f}")
    plt.legend()
    plt.grid()
    plt.show()
    return c_mean, s_mean, w_mean

class ScenarioMetricsWriter:
    """
    Per-scenario CSV with multiple algorithms.
    Columns: [scenario, algo, snr_db, C_mean, C_svd, S, WI_mean, WI_svd]

    Usage:
        writer = ScenarioMetricsWriter(scenario="LC-Gauss", outdir="outputs")
        writer.append(algo="CQKF-sw", snr_db=10,
                      C_mean=..., C_svd=..., S=..., WI_mean=..., WI_svd=...)
    """
    def __init__(self, scenario: str, outdir: str = "outputs", prefix: str = "metrics"):
        self.scenario = scenario
        self.outdir = outdir
        self.prefix = prefix
        self.fieldnames = ["scenario", "algo", "snr_db", "C_mean", "C_svd", "S", "WI_mean", "WI_svd"]

        os.makedirs(self.outdir, exist_ok=True)
        safe_scenario = re.sub(r"[^A-Za-z0-9._-]+", "_", scenario)
        self.path = os.path.join(self.outdir, f"{self.prefix}_{safe_scenario}.csv")

        # Create the file with header if new/empty
        needs_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        if needs_header:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def append(self, *, algo: str, snr_db: float,
               C_mean: float, C_svd: float, S: float, WI_mean: float, WI_svd: float) -> None:
        row = {
            "scenario": self.scenario,
            "algo": algo,
            "snr_db": snr_db,
            "C_mean": C_mean,
            "C_svd": C_svd,
            "S": S,
            "WI_mean": WI_mean,
            "WI_svd": WI_svd,
        }
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)

    def filepath(self) -> str:
        return self.path