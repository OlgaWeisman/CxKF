import torch

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
import torch.nn as nn
import time
from Linear_sysmdl import SystemModel
from Extended_data_cp import DataGen, DataLoader, DataLoader_GPU, Decimate_and_perturbate_Data, Short_Traj_Split
from Extended_data_cp import N_E, N_CV, N_T, J_T, F, H, F_rotated, H_rotated, T, T_test, m1_0, m2_0, m, n
from Pipeline_KF_CP_BNN import Pipeline_KF
from BNN_linear_KalmanNet_nn import KalmanNetNN
from datetime import datetime
from CP_test import compute_gaussian_quantiles, clalibration_residual_quantiles
path_model = 'Simulations/Linear_canonical/r=1_q=1'
import sys
sys.path.insert(1, path_model)
from model import f, h, fInacc, hInacc
from parameters import m, n, m1x_0, m2x_0, T, Q, R, Q_mod, R_mod, sigma_r, sigma_q, F, H, F_mod, H_mod
T = 10
T_test = 10
alpha = 0.05

from KalmanFilter_test import KFTest

from Plot import Plot_RTS as Plot

if 0: #torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')
    print("Running on the CPU")

print("Pipeline Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)
path_results = 'RTSNet/'

####################
### Design Model ###
####################
r2 = torch.tensor([1.])
vdB = -20  # ratio v=q2/r2
v = 10 ** (vdB / 10)
q2 = torch.mul(v, r2)

for index in range(0, len(r2)):
    print("1/r2 [dB]: ", 10 * torch.log10(1 / r2[index]))
    print("1/q2 [dB]: ", 10 * torch.log10(1 / q2[index]))

    # True model
    r = torch.sqrt(r2[index])
    q = torch.sqrt(q2[index])
    sys_model = SystemModel(F, q, H, r, T, T_test)
    sys_model.InitSequence(m1x_0, m2x_0)

    # Mismatched model
    sys_model_partialh = SystemModel(F, q, H_rotated, r, T, T_test)
    sys_model_partialh.InitSequence(m1x_0, m2x_0)

    ###################################
    ### Data Loader (Generate Data) ###
    ###################################
    dataFolderName = 'Simulations/Linear_canonical/r=1_q=1' + '/'
    dataFileName = ['1x1_rq-1010_T100_nonG.pt', '1x1_rq020_T100_nonG.pt', '1x1_rq3050_T100_nonG.pt']
    print("Start Data Gen")
    DataGen(sys_model, dataFolderName + dataFileName[index], T, T_test,randomInit=False)
    print("Data Load")
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(
        dataFolderName + dataFileName[index])
    print("trainset size:", train_target.size())
    print("cvset size:", cv_target.size())
    print("testset size:", test_target.size())

    # ##############################
    # ### Evaluate Kalman Filter ###
    # ##############################
    # print("Evaluate Kalman Filter True")
    # [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(sys_model, test_input, test_target)
    # print("Evaluate Kalman Filter Partial")
    # [MSE_KF_linear_arr_partialh, MSE_KF_linear_avg_partialh, MSE_KF_dB_avg_partialh] = KFTest(sys_model_partialh, test_input, test_target)
    #
    #
    #
    # DatafolderName = 'Filters/Linear' + '/'
    # DataResultName = 'KF_'+ dataFileName[index]
    # torch.save({
    #             'MSE_KF_linear_arr': MSE_KF_linear_arr,
    #             'MSE_KF_dB_avg': MSE_KF_dB_avg,
    #             'MSE_KF_linear_arr_partialh': MSE_KF_linear_arr_partialh,
    #             'MSE_KF_dB_avg_partialh': MSE_KF_dB_avg_partialh,
    #             }, DatafolderName+DataResultName)

    ##################
    ###  KalmanNet ###
    ##################
    test_input_KF = test_input.unsqueeze(1)
    test_target_KF = test_target.unsqueeze(1)
    [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(sys_model, test_input_KF, test_target_KF)
    # Insert new unsqeeze to the 1D dimetion KalmanNet
    train_input = train_input.unsqueeze(1).cpu()
    train_target = train_target.unsqueeze(1).cpu()
    cv_input = cv_input.unsqueeze(1).cpu()
    cv_target = cv_target.unsqueeze(1).cpu()
    test_input = test_input.unsqueeze(1).cpu()
    test_target = test_target.unsqueeze(1).cpu()
    print("Start KNet pipeline")
    print("KNet with full model info")
    modelFolder = 'KNet' + '/'
    KNet_Pipeline = Pipeline_KF(strTime, "KNet", "KNet_" + dataFileName[index])
    KNet_Pipeline.setssModel(sys_model)
    KNet_model = KalmanNetNN()
    KNet_model.Build(sys_model)
    KNet_Pipeline.setModel(KNet_model)
    KNet_Pipeline.setTrainingParams(n_Epochs=500, n_Batch=10, learningRate=1E-4, weightDecay=1E-5,num_of_mc_iterations = 20)

    # KNet_Pipeline.model = torch.load(modelFolder+"model_KNet.pt")

    KNet_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target)


    KNet_Pipeline.NNCalibrate_Predict(N_E, 5, test_input, test_target, alpha)


    # # Calculate loss
    # KNet_Pipeline.NNTest(N_T, test_input, test_target)
    # KNet_Pipeline.save()
    # [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(sys_model, test_input, test_target)
    # # Print MSE Cross Validation
    # str = "KF" + "-" + "MSE Test:"
    # print(str, MSE_KF_dB_avg, "[dB]")

    # print("KNet with partial model info")
    # modelFolder = 'KNet' + '/'
    # KNet_Pipeline = Pipeline_KF(strTime, "KNet", "KNetPartial_" + dataFileName[index])
    # KNet_Pipeline.setssModel(sys_model_partialh)
    # KNet_model = KalmanNetNN()
    # KNet_model.Build(sys_model_partialh)
    # KNet_Pipeline.setModel(KNet_model)
    # KNet_Pipeline.setTrainingParams(n_Epochs=500, n_Batch=30, learningRate=1E-3, weightDecay=1E-5)
    #
    # KNet_Pipeline.model = torch.load(modelFolder+"model_KNet.pt")
    # # KNet_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target)
    # KNet_Pipeline.NNPredict(N_T, test_input, test_target)
    # KNet_Pipeline.save()


