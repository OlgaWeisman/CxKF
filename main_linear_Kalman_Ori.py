import torch

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
import torch.nn as nn
import time
from Linear_sysmdl import SystemModel
from Extended_data import DataGen, DataLoader, DataLoader_GPU, Decimate_and_perturbate_Data, Short_Traj_Split
from Extended_data import N_E, N_CV, N_T, F, H, F_rotated, H_rotated, T, T_test, m1_0, m2_0, m, n
from Pipeline_KF import Pipeline_KF
from KalmanNet_nn import KalmanNetNN
from datetime import datetime


q2 = 0.01
r2 = 0.1
F = torch.tensor([[0.63, 0.0021],[0.0021, 1.0299]])
F_false= torch.tensor([[0.83, 0.2], [0.2, 0.83]])
H = torch.tensor([[1., 1.], [0.25, 1.]])
m2_0 = torch.tensor([[1., 0.],[0., 1.]])
m1_0 = torch.tensor([[0.5], [0.5]])
from KalmanFilter_test import KFTest

from Plot import Plot_RTS as Plot

if 0:#torch.cuda.is_available():
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
r2 = torch.tensor([0.1])
vdB = -20  # ratio v=q2/r2
v = 10 ** (vdB / 10)
q2 = torch.tensor([0.01])#torch.mul(v, r2)

for index in range(0, len(r2)):
    print("1/r2 [dB]: ", 10 * torch.log10(1 / r2[index]))
    print("1/q2 [dB]: ", 10 * torch.log10(1 / q2[index]))

    # True model
    r = torch.sqrt(r2[index])
    q = torch.sqrt(q2[index])
    sys_model = SystemModel(F, q, H, r, T, T_test)
    sys_model.InitSequence(m1_0, m2_0)

    # Mismatched model
    sys_model_partialh = SystemModel(F_false, q, H, r, T, T_test)
    sys_model_partialh.InitSequence(m1_0, m2_0)

    ###################################
    ### Data Loader (Generate Data) ###
    ###################################
    dataFolderName = 'Simulations/Linear_canonical/H=I' + '/'
    dataFileName = ['2x2_rq-1010_T100.pt', '2x2_rq020_T100.pt', '2x2_rq1030_T100.pt', '2x2_rq2040_T100.pt',
                    '2x2_rq3050_T100.pt']
    # print("Start Data Gen")
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
    print("Evaluate Kalman Filter True")
    [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(sys_model_partialh, test_input, test_target)



