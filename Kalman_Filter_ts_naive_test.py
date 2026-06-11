import torch
import torch.nn as nn
import time
import numpy as np
import math
from scipy.stats import norm
from Linear_KF_vF import KalmanFilter
from Extended_data import N_T
import sys
sys.path.append(r"C:\Users\owner\Documents\PythonCode\timeParamCPScores-master\timeParamCPScores-master\code")

from gurobipyTutorial import optimzeTimeAlphasKKTNoMaxLowerBound, optimzeTimeAlphasKKT, optimzeTimeAlphasKKTNoMaxLowerBoundMinArea


def KFTest_naive(SysModel, train_input, train_target, cv_input, cv_target, test_input, test_target):
    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_KF_linear_arr = torch.empty(N_T)

    start = time.time()
    KF = KalmanFilter(SysModel)
    KF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)
    x_array = torch.empty(train_input.shape[0], KF.T_test)
    for j in range(0, train_input.shape[0]):
        if SysModel.m == 1 & SysModel.n == 1:
            KF.GenerateSequence(train_input[j, :], KF.T_test)
        else:
            KF.GenerateSequence(train_input[j, :, :], KF.T_test)

        x_array[j,:] = KF.x

        # MSE_KF_linear_arr[j] = loss_fn(KF.x, train_target[j, :, :]).item()
        # MSE_KF_linear_arr[j] = loss_fn(test_input[j, :, :], test_target[j, :, :]).item()
    # Estimate μ and σ from the data
    mu_hat = x_array.mean(dim=0)  # np.mean(KF.x)
    sigma_hat = x_array.std(dim=0)  # ddof=0 for MLE estimate
    # Desired cumulative probability (alpha)
    alpha = 0.95

    # Compute x such that P(X <= x) = alpha
    hat_q_low = norm.ppf(0.05, loc=mu_hat, scale=sigma_hat)
    hat_q_high = norm.ppf(0.95, loc=mu_hat, scale=sigma_hat)

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

    cv_target_list = cv_target.squeeze().tolist()
    x_array_cv_list = x_array_cv.tolist()
    # alphas calibration
    R_vals = [[math.sqrt((cv_target_list[i][j] - x_array_cv_list[i][j]) ** 2 ) for j in range(len(cv_target_list[i]))] for i in range(len(cv_target_list))]

    m = optimzeTimeAlphasKKTNoMaxLowerBound(R_vals, 0.05, 100000)
    #m = optimzeTimeAlphasKKT(R_vals, 0.05, 100000)
    alphas = []
    for v in m.getVars():
        if "alphas" in v.varName:
            alphas.append(v.x)
        if "q" in v.varName:
            # print(v.x)
            print("obj: " + str(v.x))

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
    # #convert to list
    test_target_list = test_target.squeeze().tolist()
    x_array_test_list = x_array_test.tolist()
    # calculate region
    R_vals = [
        max([alphas[j] * math.sqrt((test_target_list[i][j] - x_array_test_list[i][j]) ** 2 ) for j in range(len(test_target_list[i]))]) for i in range(len(test_target_list))]


    R_vals.sort()
    R_vals.append(max(R_vals))

    ind_to_ret = math.ceil(len(R_vals) * (1-0.05))
    D_cp = R_vals[ind_to_ret]
    D_cp_all = [D_cp / a for a in alphas]
    intervals = np.zeros((KF.T_test, 2))

    D_cp_all = np.ones(KF.T_test)*D_cp_all

    # allowed to import graphics

    indices = torch.randperm(test_target.size(0))  # generate shuffled indices
    y_test_sorted = test_target[indices].squeeze()
    hat_y_test_sorted = x_array_test[indices]
    intervals[:, 0] = -D_cp_all/2
    intervals[:, 1] = D_cp_all/2
    y_lower = intervals[:, 0]
    y_upper = intervals[:, 1]
    upper_sorted = y_upper
    lower_sorted = y_lower
    interval = D_cp_all

    # import matplotlib.pyplot as plt
    #
    #
    # for i in range(KF.T_test):
    #     plt.plot(y_test_sorted[i,:], "bo")
    # plt.fill_between(
    #     np.arange(len(upper_sorted)), lower_sorted, upper_sorted, alpha=0.2, color="r",
    #     label="Pred. interval")
    # plt.xlabel("Sequence")
    # plt.ylabel("Values and prediction intervals")
    # plt.title("CQR")
    # plt.show()
    percentage = np.empty([test_target.shape[2],1])
    for jj in range(test_target.shape[2]):
        outside = np.sum((test_target[:,0,jj].numpy() < x_array_test[:,jj].numpy() + intervals[jj, 0] ) | (test_target[:,0,jj].numpy() > x_array_test[:,jj].numpy() + intervals[jj, 1]),0)
        percentage[jj] = (outside / test_target.shape[0]) * 100
    # plt.plot(percentage)
    # plt.xlabel("Sequence")
    # plt.title("CQR + time seq. : error %")
    # plt.show()
    #
    # plt.plot(interval)
    # plt.xlabel("Sequence")
    # plt.title("CQR + time seq.: interval")
    # plt.show()
    return [percentage, interval]
   #
   #  MSE_KF_linear_avg = torch.mean(MSE_KF_linear_arr)
   #  MSE_KF_dB_avg = 10 * torch.log10(MSE_KF_linear_avg)
   #
   #  # Standard deviation
   #  MSE_KF_dB_std = torch.std(MSE_KF_linear_arr, unbiased=True)
   #  MSE_KF_dB_std = 10 * torch.log10(MSE_KF_dB_std)
   #
   #  print("Kalman Filter - MSE LOSS:", MSE_KF_dB_avg, "[dB]")
   #  print("EKF - MSE STD:", MSE_KF_dB_std, "[dB]")
   #  # Print Run Time
   #  print("Inference Time:", t)
   #
   #  return [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg]



