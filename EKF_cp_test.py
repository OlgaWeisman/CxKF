import torch.nn as nn
import torch
import time
import numpy as np
from scipy.stats import norm
from EKF import ExtendedKalmanFilter


def EKFTest_cp(SysModel, train_input, train_target, cv_input, cv_target, test_input, test_target, modelKnowledge='full'):
    N_T = train_input.size()[0]

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_EKF_linear_arr = torch.empty(N_T)

    EKF = ExtendedKalmanFilter(SysModel, modelKnowledge)
    EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

    x_array = torch.empty(train_input.shape[0], EKF.T_test)
    sigma_array = torch.empty(train_input.shape[0], EKF.T_test)
    for j in range(0, N_T):
        EKF.GenerateSequence(train_input[j, :, :], EKF.T_test)
        x_array[j, :] = EKF.x
        sigma_array[j, :] = EKF.sigma.squeeze()
    # Estimate μ and σ from the data
    mu_hat = x_array.mean(dim=0)  # np.mean(KF.x)
    sigma_hat = np.sqrt(sigma_array.mean(dim=0))
    alpha = 0.95

    # Compute x such that P(X <= x) = alpha
    hat_q_low = norm.ppf(0.05, loc=mu_hat, scale=sigma_hat)
    hat_q_high = norm.ppf(0.95, loc=mu_hat, scale=sigma_hat)
    # calibration per time sequnce
    EKF = ExtendedKalmanFilter(SysModel, modelKnowledge)
    EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)
    x_array_cv = torch.empty(cv_input.shape[0], EKF.T_test)
    for j in range(0, cv_input.shape[0]):
        EKF.GenerateSequence(test_input[j, :, :], EKF.T_test)

        x_array_cv[j,:] = EKF.x

    x_lower = np.tile(hat_q_low, (x_array_cv.shape[0], 1))
    x_upper = np.tile(hat_q_high, (x_array_cv.shape[0], 1))
    error_low = x_lower - cv_target.squeeze().numpy()
    error_high = cv_target.squeeze().numpy() - x_upper
    err = np.maximum(error_high, error_low)
    cal_scores = {0: np.sort(err, 0)[::-1]}

    ## Test
    EKF = ExtendedKalmanFilter(SysModel, modelKnowledge)
    EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)
    x_array_test = torch.empty(test_input.shape[0], EKF.T_test)
    for j in range(0, test_input.shape[0]):
        EKF.GenerateSequence(test_input[j, :, :], EKF.T_test)

        x_array_test[j,:] = EKF.x

    intervals = np.zeros((EKF.T_test, 2))
    nc = np.sort(cal_scores[0], 0)

    index = int(np.ceil((1 - (0.05)) * (nc.shape[0] + 1))) - 1
    index = min(max(index, 0), nc.shape[0] - 1)
    test_err = np.vstack([nc[index, :], nc[index, :]])
    intervals[:, 0] = hat_q_low - test_err[0, :]
    intervals[:, 1] = hat_q_high + test_err[1, :]


    y_lower = intervals[:, 1]
    y_upper = intervals[:, 0]
    # allowed to import graphics
    import matplotlib.pyplot as plt

    interval = intervals[:, 1] - intervals[:, 0]
    upper_sorted = y_upper
    lower_sorted = y_lower

    for i in range(EKF.T_test):
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




