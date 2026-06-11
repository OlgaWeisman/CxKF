import torch
from EKF import ExtendedKalmanFilter
import torch.nn as nn


def EKFTest(SysModel, test_input, test_target):
    N_T = test_target.size()[0]

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_EKF_linear_arr = torch.empty(N_T)

    EKF = ExtendedKalmanFilter(SysModel)
    EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)
    x_array = torch.empty(test_input.shape[0],test_input.shape[1], EKF.T_test)
    sigma_array = torch.empty(test_input.shape[0],test_input.shape[1],test_input.shape[1], EKF.T_test)
    for j in range(0, test_input.shape[0]):
        if test_input.ndim == 3:
            EKF.GenerateSequence(test_input[j, :, :], EKF.T_test) #OLGA
            MSE_EKF_linear_arr[j] = loss_fn(EKF.x, test_target[j, :, :]).item()
        elif test_input.ndim == 4:
            EKF.GenerateSequence(test_input[j, :, :, :], EKF.T_test)
            MSE_EKF_linear_arr[j] = loss_fn(EKF.x, test_target[j, :, :, :]).item()
        x_array[j,:, :] = EKF.x
        sigma_array[j, :, :, :] = EKF.sigma###.squeeze()

    MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr)
    MSE_EKF_dB_avg = 10 * torch.log10(MSE_EKF_linear_avg)

    # Standard deviation
    MSE_EKF_dB_std = torch.std(MSE_EKF_linear_arr, unbiased=True)
    MSE_EKF_dB_std = 10 * torch.log10(MSE_EKF_dB_std)

    print("EKF - MSE LOSS:", MSE_EKF_dB_avg, "[dB]")
    print("EKF - MSE STD:", MSE_EKF_dB_std, "[dB]")


    return [x_array, sigma_array]