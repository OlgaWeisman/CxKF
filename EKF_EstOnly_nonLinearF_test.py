import torch
from EKF_varF import ExtendedKalmanFilter


def EKFTest(SysModel, test_input):

    EKF = ExtendedKalmanFilter(SysModel)
    EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)
    x_array = torch.empty(test_input.shape[0], EKF.T_test)
    sigma_array = torch.empty(test_input.shape[0], EKF.T_test)
    for j in range(0, test_input.shape[0]):
        EKF.GenerateSequence(test_input[j, :, :], EKF.T_test)

        x_array[j, :] = EKF.x
        sigma_array[j, :] = EKF.sigma.squeeze()


    return [x_array, sigma_array]