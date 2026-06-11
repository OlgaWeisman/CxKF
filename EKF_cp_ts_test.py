import torch
import torch.nn as nn
import time
import numpy as np
import math
from scipy.stats import norm
from EKF import ExtendedKalmanFilter
from Extended_data import N_T
import sys
sys.path.append(r"C:\Users\owner\Documents\PythonCode\timeParamCPScores-master\timeParamCPScores-master\code")

from gurobipyTutorial import optimzeTimeAlphasKKTNoMaxLowerBound, optimzeTimeAlphasKKT


def EKFTest(SysModel, train_input, train_target, cv_input, cv_target, test_input, test_target,first_method):
    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_KF_linear_arr = torch.empty(N_T)

    start = time.time()
    EKF = ExtendedKalmanFilter(SysModel)
    EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)
    x_array = torch.empty(train_input.shape[0], EKF.T_test)
    sigma_array = torch.empty(train_input.shape[0], EKF.T_test)
    for j in range(0, train_input.shape[0]):
        EKF.GenerateSequence(train_input[j, :, :], EKF.T_test)

        x_array[j,:] = EKF.x
        sigma_array[j, :] = EKF.sigma.squeeze()

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

    # calibration per time sequnce
    EKF = ExtendedKalmanFilter(SysModel)
    EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)
    x_array_cv = torch.empty(cv_input.shape[0], EKF.T_test)
    for j in range(0, cv_input.shape[0]):
        if SysModel.m == 1 & SysModel.n == 1:
            EKF.GenerateSequence(cv_input[j,:, :], EKF.T_test)
        else:
            EKF.GenerateSequence(cv_input[j, :, :], EKF.T_test)

        x_array_cv[j,:] = EKF.x
    if first_method:
        print("Union")
        #TODO

    else:
        x_lower = np.tile(hat_q_low, (x_array_cv.shape[0], 1))
        x_upper = np.tile(hat_q_high, (x_array_cv.shape[0], 1))
        error_low = x_lower - cv_target.squeeze().numpy()
        error_high = cv_target.squeeze().numpy() - x_upper
        err = np.abs(np.maximum(error_high, error_low))
        R_vals = err.tolist()
        m = optimzeTimeAlphasKKTNoMaxLowerBound(R_vals, 0.05, 100000)
        alphas = []
        for v in m.getVars():
            if "alphas" in v.varName:
                alphas.append(v.x)
            if "q" in v.varName:
                # print(v.x)
                print("obj: " + str(v.x))



    ## Test
    EKF = ExtendedKalmanFilter(SysModel)
    EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)
    x_array_test = torch.empty(test_input.shape[0], EKF.T_test)
    for j in range(0, test_input.shape[0]):
        if SysModel.m == 1 & SysModel.n == 1:
            EKF.GenerateSequence(test_input[j, :], EKF.T_test)
        else:
            EKF.GenerateSequence(test_input[j, :, :], EKF.T_test)

        x_array_test[j,:] = EKF.x
    # # #convert to list
    if first_method:
        x_lower = np.tile(hat_q_low, (test_target.shape[0], 1))
        x_upper = np.tile(hat_q_high, (test_target.shape[0], 1))
        error_low = x_lower - test_target.squeeze().numpy()
        error_high = test_target.squeeze().numpy() - x_upper
        err = np.maximum(error_high, error_low)
        alpha_err = np.max(err/EKF.T_test, axis=1)
        cal_scores = {0: np.sort(alpha_err, 0)[::-1]}
        nc = np.sort(cal_scores[0], 0)
        index = int(np.ceil((alpha) * (nc.shape[0] + 1))) - 1
        index = min(max(index, 0), nc.shape[0] - 1)
        D_cp = nc[index]/2
        D_cp_all = D_cp * EKF.T_test
    # #convert to list
    else:
        x_lower = np.tile(hat_q_low, (test_target.shape[0], 1))
        x_upper = np.tile(hat_q_high, (test_target.shape[0], 1))
        error_low = x_lower - test_target.squeeze().numpy()
        error_high = test_target.squeeze().numpy() - x_upper
        err = np.maximum(error_high, error_low)
        alpha_err = np.max(err*alphas, axis=1)
        cal_scores = {0: np.sort(alpha_err, 0)[::-1]}
        nc = np.sort(cal_scores[0], 0)
        index = int(np.ceil((alpha) * (nc.shape[0] + 1))) - 1
        index = min(max(index, 0), nc.shape[0] - 1)
        D_cp = nc[index]/2
        D_cp_all = [D_cp / a for a in alphas]

    intervals = np.zeros((EKF.T_test, 2))

    intervals[:, 0] = hat_q_low - D_cp_all
    intervals[:, 1] = hat_q_high + D_cp_all
    y_lower = intervals[:, 1]
    y_upper = intervals[:, 0]
    # allowed to import graphics
    import matplotlib.pyplot as plt

    interval = intervals[:, 1] - intervals[:, 0]
    # sort_ind = np.argsort(interval)

    indices = torch.randperm(test_target.size(0))  # generate shuffled indices
    y_test_sorted = test_target[indices]
    upper_sorted = y_upper
    lower_sorted = y_lower
    mean = (upper_sorted + lower_sorted) / 2

    # Center such that the mean of the prediction interval is at 0.0
    # y_test_sorted -= mean
    # upper_sorted -= mean
    # lower_sorted -= mean
    for i in range(EKF.T_test):
        plt.plot(y_test_sorted[i,0,:], "bo")
    # plt.plot(y_test_sorted[6, 0, :], "ro")
    plt.fill_between(
        np.arange(len(upper_sorted)), lower_sorted, upper_sorted, alpha=0.2, color="r",
        label="Pred. interval")
    plt.xlabel("Sequence")
    plt.ylabel("Values and prediction intervals")
    plt.title("CQR + time seq")
    plt.show()
    percentage = np.empty([y_test_sorted.shape[2],1])
    for jj in range(y_test_sorted.shape[2]):
        outside = np.sum((y_test_sorted[:, 0,jj].numpy() < intervals[jj, 0] ) | (y_test_sorted[:,0,jj].numpy() > intervals[jj, 1]),0)
        percentage[jj] = (outside / y_test_sorted.shape[0]) * 100
    end = time.time()
    t = end - start

    return [percentage, interval]



