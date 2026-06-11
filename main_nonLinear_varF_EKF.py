import torch

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
import torch.nn as nn
import time
from Extended_sysmdl_varF import SystemModel
from Extended_data_cp import DataGen, DataLoader, DataLoader_GPU, Decimate_and_perturbate_Data, Short_Traj_Split
from Extended_data_cp import N_E, N_CV, N_T, J_T, F, H, F_rotated, H_rotated, T, T_test, m1_0, m2_0, m, n
from EKF_EstOnly_nonLinearF_test import EKFTest
from CP_test import compute_gaussian_quantiles, clalibration_residual_quantiles
from datetime import datetime
path_model = 'Simulations/nonLinear_varF_1D'
import sys
sys.path.insert(1,path_model)

from model import f, h
from parameters import T, m1x_0, m2x_0, m, n
T = 100
T_test = 100
if False: #torch.cuda.is_available():
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
r2 = torch.tensor([10, 1.,1e-3])
vdB = -20  # ratio v=q2/r2
v = 10 ** (vdB / 10)
q2 = torch.mul(v, r2)
alpha = 0.05
for index in range(0, len(r2)):
    print("1/r2 [dB]: ", 10 * torch.log10(1 / r2[index]))
    print("1/q2 [dB]: ", 10 * torch.log10(1 / q2[index]))

    # True model
    r = torch.sqrt(r2[index])
    q = torch.sqrt(q2[index])
    sys_model = SystemModel(f, q, h, r, T, T, m, n, "1D")  # arbitary q and r
    sys_model.InitSequence(m1x_0, m2x_0)


    ###################################
    ### Data Loader (Generate Data) ###
    ###################################
    dataFolderName = 'Simulations/Linear_canonical/r=1_q=1' + '/'
    dataFileName = ['1x1_rq-1010_T100.pt', '1x1_rq020_T100.pt', '1x1_rq1030_T100.pt', '1x1_rq2040_T100.pt',
                    '1x1_rq3050_T100.pt']
    print("Start Data Gen")
    DataGen(sys_model, dataFolderName + dataFileName[index], T, T_test,randomInit=False)
    print("Data Load")
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader(
        dataFolderName + dataFileName[index])
    print("trainset size:", train_target.size())
    print("cvset size:", cv_target.size())
    print("testset size:", test_target.size())
    print("Evaluate Kalman Filter True")
    [x_array_train, sigma_array_train] = EKFTest(sys_model, train_input)
    [q_low, q_hi, q_half] = compute_gaussian_quantiles(x_array_train, sigma_array_train, alpha)
    [q_low_aT, q_hi_aT, q_half_aT] = compute_gaussian_quantiles(x_array_train, sigma_array_train, alpha/T)
    title_value = 10 * torch.log10(1 / r2[index]).item()
    [c, w, s] = clalibration_residual_quantiles(x_array_train, train_target, q_low, q_hi, J_T, alpha, title_value,q_half, q_low_aT, q_hi_aT,title_value)






