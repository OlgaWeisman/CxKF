import torch

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
import torch.nn as nn
import time
from Linear_sysmdl_Fv import SystemModel
from Extended_data import DataGen, DataLoader, DataLoader_GPU, Decimate_and_perturbate_Data, Short_Traj_Split
from Extended_data import N_E, N_CV, N_T, F, H, F_rotated, H_rotated, T, T_test, m1_0, m2_0, m, n
from Kalman_Filter_cp_ts_test import KFTest
from KalmanFilter_cp_test import KFTest_cp
from Kalman_Filter_ts_naive_test import KFTest_naive
from datetime import datetime
import matplotlib.pyplot as plt
path_model = 'Simulations/Linear_canonical/r=1_q=1'
import sys
sys.path.insert(1, path_model)
from model import f, h, fInacc, hInacc
from parameters import m, n, m1x_0, m2x_0, T, Q, R, Q_mod, R_mod, sigma_r, sigma_q, F, H, F_mod, H_mod
T = 20
T_test = 20
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

    # ##############################
    # ### Evaluate Kalman Filter ###,P:L 
    # ##############################
    print("Evaluate Kalman Filter True")
    [pr_cp_ts, interval_cp_ts] = KFTest(sys_model, train_input, train_target, cv_input[:, :,:], cv_target[:, :,:], test_input, test_target,True)
    [pr_cp_ts_mt2, interval_cp_ts_mt2] = KFTest(sys_model, train_input, train_target, cv_input[0:200, :, :], cv_target[0:200, :, :], test_input, test_target,False)
    [pr_cp_naive, interval_naive] = KFTest_naive(sys_model, train_input, train_target, cv_input[0:200, :, :],  cv_target[0:200, :, :], test_input, test_target)
    [pr_cqr, interval_cqr] = KFTest_cp(sys_model, train_input, train_target, cv_input, cv_target, test_input, test_target,False )
    [pr_cqr_go, interval_cqr_go] = KFTest_cp(sys_model, train_input, train_target, cv_input, cv_target, test_input,
                                       test_target, True)


    title_value = 10 * torch.log10(1 / r2[index]).item()
    # plt.plot(nc,label = 'Union')
    # plt.plot(nc2, label = 'LCP')
    # plt.title(f"1/r2 [dB]: {title_value:.2f}")
    # plt.legend()
    # plt.show()

    mean_pr_cp_ts = pr_cp_ts.mean()
    plt.plot(pr_cp_ts, label=f'cqr + time-series Union (mean={mean_pr_cp_ts:.3f})')

    mean_pr_cp_ts_mt2 = pr_cp_ts_mt2.mean()
    plt.plot(pr_cp_ts_mt2, label=f'cqr + time-series (mean={mean_pr_cp_ts_mt2:.3f})')
    mean_pr_cp_naive = pr_cp_naive.mean()
    plt.plot(pr_cp_naive, label=f'conformal-time-series (mean={mean_pr_cp_naive:.3f})')
    mean_pr_cqr = pr_cqr.mean()
    plt.plot(pr_cqr, label=f'cqr (mean={mean_pr_cqr:.3f})')
    mean_pr_cqr_go = pr_cqr_go.mean()
    plt.plot(pr_cqr_go, label=f'Gaussian only (mean={mean_pr_cqr_go:.3f})')
    plt.xlabel("Sequence")
    plt.ylabel("Error %")
    plt.title(f"1/r2 [dB]: {title_value:.2f}")
    plt.legend()
    plt.show()

    mean_interval_cp_ts = interval_cp_ts.mean()
    plt.plot(interval_cp_ts, label=f'cqr + time-series Union (mean={mean_interval_cp_ts:.3f})')
    mean_interval_cp_ts_mt2 = interval_cp_ts_mt2.mean()
    plt.plot(interval_cp_ts_mt2, label=f'cqr + time-series (mean={mean_interval_cp_ts_mt2:.3f})')
    mean_interval_naive = interval_naive.mean()
    plt.plot(interval_naive, label=f'conformal-time-series (mean={mean_interval_naive:.3f})')
    mean_interval_cqr = interval_cqr.mean()
    plt.plot(interval_cqr, label=f'cqr (mean={mean_interval_cqr:.3f})')
    plt.xlabel("Sequence")
    plt.ylabel("Interval")
    plt.title(f"1/r2 [dB]: {title_value:.2f}")
    plt.legend()
    plt.show()





