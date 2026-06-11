import torch
import torch.nn as nn
import time
from Linear_KF import KalmanFilter
from Extended_data import N_T


def KFTest(SysModel, test_input):

    KF = KalmanFilter(SysModel)
    KF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)
    if test_input.shape[1] > 1 and test_input.ndim > 2:
        x_array = torch.empty(test_input.shape[0],test_input.shape[1],KF.T_test)
        sigma_array = torch.empty(test_input.shape[0],test_input.shape[1],test_input.shape[1], KF.T_test)
    else:
        x_array = torch.empty(test_input.shape[0], KF.T_test)
        sigma_array = torch.empty(test_input.shape[0], KF.T_test)
    for j in range(0, test_input.shape[0]):
        KF.GenerateSequence(test_input[j, :, :], KF.T_test)

        x_array[j, :] = KF.x
        sigma_array[j, :] = KF.sigma.squeeze()


    return [x_array, sigma_array]



