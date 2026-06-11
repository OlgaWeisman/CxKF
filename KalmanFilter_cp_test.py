import torch
import torch.nn as nn
import time
import numpy as np
from scipy.stats import norm
from Linear_KF_vF import KalmanFilter
from Extended_data import N_T


def KFTest_cp(SysModel, train_input, train_target, cv_input, cv_target, test_input, test_target,use_gaussian_pdfs_only):
    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_KF_linear_arr = torch.empty(N_T)

    start = time.time()
    KF = KalmanFilter(SysModel)
    KF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)
    x_array = torch.empty(train_input.shape[0], KF.T_test)
    sigma_array = torch.empty(train_input.shape[0], KF.T_test)
    for j in range(0, train_input.shape[0]):
        if SysModel.m == 1 & SysModel.n == 1:
            KF.GenerateSequence(train_input[j, :], KF.T_test)
        else:
            KF.GenerateSequence(train_input[j, :, :], KF.T_test)

        x_array[j, :] = KF.x
        sigma_array[j, :] = KF.sigma.squeeze()
        # MSE_KF_linear_arr[j] = loss_fn(KF.x, train_target[j, :, :]).item()
        # MSE_KF_linear_arr[j] = loss_fn(test_input[j, :, :], test_target[j, :, :]).item()
    # Estimate μ and σ from the data
    mu_hat = x_array.mean(dim=0)  # np.mean(KF.x)
    sigma_hat = np.sqrt(sigma_array.mean(dim=0))
    # Desired cumulative probability (alpha)
    alpha = 0.95

    # Compute x such that P(X <= x) = alpha
    hat_q_low = norm.ppf(0.05, loc=mu_hat, scale=sigma_hat)
    hat_q_high = norm.ppf(0.95, loc=mu_hat, scale=sigma_hat)
    if ~use_gaussian_pdfs_only:
        # calibration per time sequnce
        KF = KalmanFilter(SysModel)
        KF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)
        x_array_cv = torch.empty(cv_input.shape[0], KF.T_test)
        for j in range(0, cv_input.shape[0]):
            if SysModel.m == 1 & SysModel.n == 1:
                KF.GenerateSequence(cv_input[j,:, :], KF.T_test)
            else:
                KF.GenerateSequence(cv_input[j, :, :], KF.T_test)

            x_array_cv[j,:] = KF.x

        x_lower = np.tile(hat_q_low, (x_array_cv.shape[0], 1))
        x_upper = np.tile(hat_q_high, (x_array_cv.shape[0], 1))
        error_low = x_lower - cv_target.squeeze().numpy()
        error_high = cv_target.squeeze().numpy() - x_upper
        err = np.maximum(error_high, error_low)
        cal_scores = {0: np.sort(err, 0)[::-1]}

        ## Test
        KF = KalmanFilter(SysModel)
        KF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)
        x_array_test = torch.empty(test_input.shape[0], KF.T_test)
        for j in range(0, test_input.shape[0]):
            if SysModel.m == 1 & SysModel.n == 1:
                KF.GenerateSequence(test_input[j, :], KF.T_test)
            else:
                KF.GenerateSequence(test_input[j, :, :], KF.T_test)

            x_array_test[j,:] = KF.x

        intervals = np.zeros((KF.T_test, 2))
        nc = np.sort(cal_scores[0], 0)
        index = int(np.ceil((1 - (0.05)) * (nc.shape[0] + 1))) - 1
        index = min(max(index, 0), nc.shape[0] - 1)
        test_err = np.vstack([nc[index, :], nc[index, :]])
        intervals[:, 0] = hat_q_low - test_err[0, :]
        intervals[:, 1] = hat_q_high + test_err[1, :]
    else:
        intervals = np.zeros((KF.T_test, 2))
        intervals[:, 0] = hat_q_low
        intervals[:, 1] = hat_q_high

    y_lower = intervals[:, 1]
    y_upper = intervals[:, 0]
    # allowed to import graphics
    import matplotlib.pyplot as plt

    interval = intervals[:, 1] - intervals[:, 0]
    upper_sorted = y_upper
    lower_sorted = y_lower

    for i in range(KF.T_test):
        plt.plot(test_target[i,0,:], "bo")
    plt.fill_between(
        np.arange(len(upper_sorted)), lower_sorted, upper_sorted, alpha=0.2, color="r",
        label="Pred. interval")
    plt.xlabel("Sequence")
    plt.ylabel("Values and prediction intervals")
    plt.title("CQR")
    plt.show()
    percentage = np.empty([test_target.shape[2],1])
    for jj in range(test_target.shape[2]):
        outside = np.sum((test_target[:,0,jj].numpy() < intervals[jj, 0] ) | (test_target[:,0,jj].numpy() > intervals[jj, 1]),0)
        percentage[jj] = (outside / test_target.shape[0]) * 100

    return [percentage, interval]

    # MSE_KF_linear_avg = torch.mean(MSE_KF_linear_arr)
    # MSE_KF_dB_avg = 10 * torch.log10(MSE_KF_linear_avg)
    #
    # # Standard deviation
    # MSE_KF_dB_std = torch.std(MSE_KF_linear_arr, unbiased=True)
    # MSE_KF_dB_std = 10 * torch.log10(MSE_KF_dB_std)
    #
    # print("Kalman Filter - MSE LOSS:", MSE_KF_dB_avg, "[dB]")
    # print("EKF - MSE STD:", MSE_KF_dB_std, "[dB]")
    # # Print Run Time
    # print("Inference Time:", t)
    #
    # return [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg]



