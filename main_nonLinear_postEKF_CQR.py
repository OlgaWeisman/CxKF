import torch

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
import torch.nn as nn
import time
from Extended_sysmdl import SystemModel
from Extended_data_cp import DataGen, DataLoader, DataLoader_GPU, Decimate_and_perturbate_Data, Short_Traj_Split
from Extended_data_cp import N_E, N_CV, N_T, J_T, F, H, F_rotated, H_rotated, T, T_test, m1_0, m2_0, m, n
from Pipeline_EKF import Pipeline_EKF
from Pipleline_post_EKF_CP import Pipeline_post_EKF
from Extended_KalmanNet_nn import KalmanNetNN
from Quantile_arch1_GRU import QuantileNN
from datetime import datetime
from CP_test import compute_gaussian_quantiles, clalibration_residual_quantiles
path_model = 'Simulations/nonLinear_1D'
import sys
sys.path.insert(1,path_model)
from model import f, h, f_tylor
from parameters import T, m1x_0, m2x_0, m, n
T = 10
T_test = 10
alpha = 0.05

from KalmanFilter_test import KFTest


if torch.cuda.is_available():
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
r2 = torch.tensor([10, 1, 1e-1])
vdB = -20  # ratio v=q2/r2
v = 10 ** (vdB / 10)
q2 = torch.mul(v, r2)

for index in range(0, len(r2)):
    print("1/r2 [dB]: ", 10 * torch.log10(1 / r2[index]))
    print("1/q2 [dB]: ", 10 * torch.log10(1 / q2[index]))

    # True model
    r = torch.sqrt(r2[index])
    q = torch.sqrt(q2[index])
    sys_model = SystemModel(f, q, h, r, T, T, m, n, "1D")  # arbitary q and r
    sys_model.InitSequence(m1x_0, m2x_0)

    sys_model_partialf = SystemModel(f_tylor, q, h, r, T, T, m, n, "1D")  # arbitary q and r
    sys_model_partialf.InitSequence(m1x_0, m2x_0)
    ###################################
    ### Data Loader (Generate Data) ###
    ###################################
    dataFolderName = 'Simulations/Linear_canonical/r=1_q=1' + '/'
    dataFileName = ['1x1_rq1010_nonLinear.pt', '1x1_rq020_nonLinear.pt', '1x1_rq1030_nonLinear.pt']
    # print("Start Data Gen")
    # DataGen(sys_model, dataFolderName + dataFileName[index], T, T_test,randomInit=False)
    # continue
    print("Data Load")
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(
        dataFolderName + dataFileName[index])
    print("trainset size:", train_target.size())
    print("cvset size:", cv_target.size())
    print("testset size:", test_target.size())
    print("Evaluate Kalman Filter True")

    # Insert new unsqeeze to the 1D dimetion KalmanNet
    train_input = train_input.unsqueeze(1)
    train_target = train_target.unsqueeze(1)
    cv_input = cv_input.unsqueeze(1)
    cv_target = cv_target.unsqueeze(1)
    test_input = test_input.unsqueeze(1)
    test_target = test_target.unsqueeze(1)
    print("Start KNet pipeline")
    print("KNet with full model info")
    modelFolder = 'KNet' + '/'
    ##############################################
    ####### KalmanNet training ###################
    ##############################################
    KF_Pipeline = Pipeline_EKF(strTime, "KNet", "KNet_" + dataFileName[index])
    KF_Pipeline.setssModel(sys_model_partialf)
    KalmanNet_model = KalmanNetNN()
    KalmanNet_model.Build(sys_model_partialf)
    KF_Pipeline.setModel(KalmanNet_model)
    KF_Pipeline.setTrainingParams(n_Epochs=500, n_Batch=10, learningRate=1E-3, weightDecay=1E-5)
    KF_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target)


    ###############################################
    ############### Quantiles training ############
    ###############################################
    Quantile_Pipeline = Pipeline_post_EKF(strTime, "KNet", "KNet_" + dataFileName[index])
    Quantile_Pipeline.setssModel(sys_model_partialf)
    Quantile_model = QuantileNN(new_head_in_shape = 4)
    Quantile_Pipeline.setModel(Quantile_model)
    Quantile_Pipeline.setTrainingParams(n_Epochs=500, n_Batch=10, learningRate=1E-3, weightDecay=1E-5)
    Quantile_Pipeline.NNTrain(KF_Pipeline.model, N_E, train_input, train_target, N_CV, cv_input, cv_target)


    Quantile_Pipeline.NNCalibrate_Predict(KF_Pipeline.model, N_E, J_T, test_input, test_target, alpha)


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


