
import torch
import math
import wandb
import copy
import os
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
import torch.nn as nn
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from Extended_sysmdl import SystemModel
from Extended_data_cp import DataGen, DataLoader, DataLoader_GPU, Decimate_and_perturbate_Data, Short_Traj_Split
from Extended_data_cp import N_E, N_CV, N_T, J_T, F, H, F_rotated, H_rotated, T, T_test, m1_0, m2_0, m, n
from EKF_EstOnly_test import EKFTest
from UKF_test import UKFTest
# from CP_md_test import mahalanobis_elliptical_region, naive_rectangular_region_with_gaussian_quantiles
from DQREstimator import DQREstimator, mahalanobis_elliptical_region, naive_rectangular_region_with_gaussian_quantiles, rectangular_no_learning
from mahalanobis_elliptical_region_no_learning import MahalanobisEllipticalRegionNoLearning
from datetime import datetime
path_model = 'Simulations/Toy_problems/'
import sys
sys.path.insert(1,path_model)

from model import f, h
from parameters import T, m1x_0, m2x_0, m, n
T = 10
T_test = 10
train_model = False
load_path = None
Epochs_num = 5
model_type = 'EKF'
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
r2 = torch.tensor([20])
vdB = -20  # ratio v=q2/r2
v = 10 ** (vdB / 10)
q2 = torch.mul(v, r2)
alpha = 0.05#0.0072
# wandb.init(
#     project="kalmannet-Toy",
#     config={
#         "alpha": alpha,
#         "vdB": vdB,
#         "T": T,
#         "T_test": T_test,
#     }
# )
for index in range(0, len(r2)):
    print("1/r2 [dB]: ", 10 * torch.log10(1 / r2[index]))
    print("1/q2 [dB]: ", 10 * torch.log10(1 / q2[index]))

    # True model
    r = torch.sqrt(r2[index])
    q = torch.sqrt(q2[index])
    sys_model = SystemModel(f, q, h, r, T, T_test, m, n, "Toy", device = dev)
    sys_model.InitSequence(m1x_0, m2x_0)


    ###################################
    ### Data Loader (Generate Data) ###
    ###################################
    dataFolderName = 'Simulations/Toy_problems/' + '/'
    dataFileName = ['1x1_rq-1010_T100.pt', '1x1_rq020_T100.pt', '1x1_rq1030_T100.pt', '1x1_rq2040_T100.pt',
                    '1x1_rq3050_T100.pt']
    if not train_model:
        print("Start Data Gen")
        DataGen(sys_model, dataFolderName + dataFileName[index], T, T_test,randomInit=False)
    print("Data Load")
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader(
        dataFolderName + dataFileName[index])
    print("trainset size:", train_target.size())
    print("cvset size:", cv_target.size())
    print("testset size:", test_target.size())
    print("Evaluate Kalman Filter True")
    file_name = (
        f"{model_type}_outputs"
        f"_T{T}"
        f"_r2{str(r2).replace('.', 'p')}"
        f"_alpha{str(alpha).replace('.', 'p')}"
        ".pt"
    )
    load_path_KF = os.path.join(path_model, file_name)
    if train_model:
        if model_type == 'UKF':
            [_,_,_,x_array_train, sigma_array_train] = UKFTest(sys_model, train_input, train_target)
            [_,_,_,x_array_cv, sigma_array_cv] = UKFTest(sys_model, cv_input, cv_target)
            [_,_,_,x_array_test, sigma_array_test] = UKFTest(sys_model, test_input, test_target)
        elif model_type == 'EKF':
            [x_array_train, sigma_array_train] = EKFTest(sys_model, train_input, train_target)
            [x_array_cv, sigma_array_cv] = EKFTest(sys_model, cv_input, cv_target)
            [x_array_test, sigma_array_test] = EKFTest(sys_model, test_input, test_target)

        torch.save({
            "x_array_train": x_array_train,
            "sigma_array_train": sigma_array_train,
            "x_array_cv": x_array_cv,
            "sigma_array_cv": sigma_array_cv,
            "x_array_test": x_array_test,
            "sigma_array_test": sigma_array_test,
        }, load_path_KF)
    else:

        print("Loading EKF outputs...")

        data = torch.load(load_path_KF)

        x_array_train = data["x_array_train"]
        sigma_array_train = data["sigma_array_train"]

        x_array_cv = data["x_array_cv"]
        sigma_array_cv = data["sigma_array_cv"]

        x_array_test = data["x_array_test"]
        sigma_array_test = data["sigma_array_test"]


    error_NPDQR = torch.empty(T, Epochs_num)
    IW_NPQR = torch.empty(T, Epochs_num)

    error_NPDQR_before_cal = torch.empty(T, Epochs_num)
    IW_NPQR_before_cal = torch.empty(T, Epochs_num)

    error_NPDQR_small = torch.empty(T, Epochs_num)
    IW_NPQR_small = torch.empty(T, Epochs_num)

    error_NPDQR_small_before_cal = torch.empty(T, Epochs_num)
    IW_NPQR_small_before_cal = torch.empty(T, Epochs_num)

    error_rect = torch.empty(T, Epochs_num)
    IW_rect = torch.empty(T, Epochs_num)

    error_rect_before_cal = torch.empty(T, Epochs_num)
    IW_rect_before_cal = torch.empty(T, Epochs_num)

    error_elip_crc = torch.empty(T, Epochs_num)
    IW_elip_crc = torch.empty(T, Epochs_num)

    error_elip_crc_before_cal = torch.empty(T, Epochs_num)
    IW_elip_crc_before_cal = torch.empty(T, Epochs_num)

    # split to calibtaion and test
    N = x_array_test.shape[0]
    x_input_train = torch.cat([x_array_train, torch.sqrt(sigma_array_train.flatten(start_dim=1, end_dim=2))], dim=1)
    x_input_cv = torch.cat([x_array_cv, torch.sqrt(sigma_array_cv.flatten(start_dim=1, end_dim=2))], dim=1)
    dqr = DQREstimator(
        hs_str="[128,128]",
        num_ep=500,
        num_u=128,
        batch_size=128,
        lr=1e-3,
        wd=1e-4,
        dropout=0.01,
        device='cpu',  # or 'cuda'
        patience=100,
        debug_plot=False
    )

    dqr_fixing = DQREstimator(
        hs_str="[128,128]",
        num_ep=500,
        num_u=128,
        batch_size=128,
        lr=1e-3,
        wd=1e-4,
        dropout=0.01,
        device='cpu',  # or 'cuda'
        patience=100,
        debug_plot=False
    )

    dqr_small = DQREstimator(
        hs_str="[128,128]",
        num_ep=500,
        num_u=16,
        batch_size=128,
        lr=1e-3,
        wd=1e-4,
        dropout=0.01,
        device='cpu',  # or 'cuda'
        patience=100,
        debug_plot=False
    )
    # keep original learned model
    dqr_learned = dqr
    # create another DQR/CRC object for elliptical model
    dqr_elip = copy.deepcopy(dqr)
    dqr_elip.num_u = None
    elliptical_model = MahalanobisEllipticalRegionNoLearning(
        dim=2,
        alpha=alpha
    )

    # create another DQR/CRC object for rectangular model
    dqr_rect = copy.deepcopy(dqr)
    dqr_rect.u_list = torch.tensor([
        [1., 0.],  # x1 negative direction  -> high
        [-1., 0.],  # x1 positive direction  -> low
        [0., 1.],  # x2 negative direction  -> high
        [0., -1.],  # x2 positive direction  -> low
    ])

    dqr_rect.model = lambda x_hat: rectangular_no_learning(
        x_hat=x_hat,
        alpha=alpha,
        d=2
    )

    dqr_elip.model = elliptical_model

    common_config = {
        "alpha": float(alpha),
        "T": int(T),
        "num_ep": dqr.num_ep,
        "num_u": dqr.num_u,
        "batch_size": dqr.batch_size,
        "lr": dqr.lr,
        "wd": dqr.wd,
        "dropout": dqr.dropout,
        "device": "cpu",
    }

    group_name = f"compare_learning_{strTime}"

    # =========================
    # First run: DQR
    # =========================
    with wandb.init(
            project="Linear Gaussian",
            name=f"DQR_{strTime}",
            group=group_name,
            job_type="DQR",
            config={**common_config, "method": "DQR"},
            reinit="finish_previous",
    ) as run:
        model_name = f"dqr_model_{model_type}_T{T}_r2{r2}_alpha{alpha:.3f}.pt"
        if not train_model: load_path = os.path.join(path_model, model_name)
        dqr.fit_nonLinear_forall_seq(
            x_input_train,
            train_target,
            x_input_cv,
            cv_target,
            tau=alpha/7,
            log_wandb=True,
            load_path=load_path
        )
        if train_model:
            torch.save(
                {
                    "model_state_dict": dqr.model.state_dict(),
                    "u_list": dqr.u_list,
                },
                os.path.join(path_model, model_name)
            )

    # =========================
    # Second run: DQR fixing
    # =========================
    with wandb.init(
            project="Linear Gaussian",
            name=f"DQR_fixing_{strTime}",
            group=group_name,
            job_type="Gaussian_correction",
            config={**common_config, "method": "Gaussian_correction"},
            reinit="finish_previous",
    ) as run:

        model_name = f"dqr_fixing_model_{model_type}_T{T}_r2{r2}_alpha{alpha:.3f}.pt"
        if not train_model: load_path = os.path.join(path_model, model_name)
        dqr_fixing.fit_gaussian_correction_forall_seq(
            x_input_train,
            train_target,
            x_input_cv,
            cv_target,
            tau=alpha/7,
            log_wandb=True,
            load_path=load_path
        )
    wandb.finish()
    if train_model:
        torch.save(
            {
                "model_state_dict": dqr_fixing.model.state_dict(),
                "u_list": dqr_fixing.u_list,
            },
            os.path.join(path_model, model_name)
        )
    model_name = f"dqr_small_model_{model_type}_T{T}_r2{r2}_alpha{alpha:.3f}.pt"
    if not train_model: load_path = os.path.join(path_model, model_name)
    dqr_small.fit_nonLinear_forall_seq(
        x_input_train,
        train_target,
        x_input_cv,
        cv_target,
        tau=alpha/7,
        log_wandb=False,
        load_path=load_path
    )
    if train_model:
        torch.save(
            {
                "model_state_dict": dqr_small.model.state_dict(),
                "u_list": dqr_small.u_list,
            },
            os.path.join(path_model, model_name)
        )

    for epochs in range(0, Epochs_num):
        indices = np.random.permutation(N)

        # 80% / 20%
        split_idx = int(0.8 * N)
        calib_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        hat_x_calib = x_array_test[calib_idx]
        target_x_calib = test_target[calib_idx]
        sigma_calib = sigma_array_test[calib_idx]

        B, D1, D2, T1 = sigma_calib.shape

        sigma_calib_flat = sigma_calib.reshape(B, D1 * D2, T1)

        hat_x_test = x_array_test[test_idx]
        target_x_test = test_target[test_idx]
        sigma_test = sigma_array_test[test_idx]

        B, D1, D2, T1 = sigma_test.shape

        sigma_test_flat = sigma_test.reshape(B, D1 * D2, T1)
        # bin_id_per_t = np.zeros((hat_x_test.shape[0], T), dtype=int)
        # plot_data = {}  # key: (t,b) -> dict of stuff to plot
        #
        # error_elip, interval_width_elip = mahalanobis_elliptical_region(target_x_test, hat_x_test, sigma_test,
        #                                                                 train_target, alpha, bin_id_per_t,
        #                                                                 plot_data=plot_data)
        # error_naive, interval_width_naive = naive_rectangular_region_with_gaussian_quantiles(target_x_test, hat_x_test,
        #                                                                                      sigma_test, train_target,
        #                                                                                      alpha, bin_id_per_t,
        #                                                                                      plot_data=plot_data)

        for t in range(0, T):
            x_input_calib_per_t = torch.cat([hat_x_calib[:, :, t], torch.sqrt(sigma_calib_flat[:, :, t])], dim=1)

            dqr.calibrate_crc(x_input_calib_per_t, target_x_calib[:, :, t], alpha)
            dqr_small.calibrate_crc(x_input_calib_per_t, target_x_calib[:, :, t], alpha)

            x_input_test_per_t = torch.cat([hat_x_test[:, :, t], torch.sqrt(sigma_test_flat[:, :, t])], dim=1)
            if t == 9 and epochs == 4:
                plot_flag = True
            else: plot_flag = False
            error_NPDQR[t, epochs], IW_NPQR[t, epochs] = dqr.inference_new(x_input_test_per_t, target_x_test[:, :, t],
                                                                           target_x_test[:, :, t],
                                                                           False,plot_flag)

            error_NPDQR_before_cal[t, epochs], IW_NPQR_before_cal[t, epochs] = dqr.inference_new(x_input_test_per_t,
                                                                                                 target_x_test[:, :, t],
                                                                                                 target_x_test[:, :, t],
                                                                                                 True,plot_flag)

            error_NPDQR_small[t, epochs], IW_NPQR_small[t, epochs] = dqr_small.inference_new(x_input_test_per_t,
                                                                                             target_x_test[:, :, t],
                                                                                             target_x_test[:, :, t],
                                                                                             False,plot_flag)

            error_NPDQR_small_before_cal[t, epochs], IW_NPQR_small_before_cal[t, epochs] = dqr_small.inference_new(
                x_input_test_per_t, target_x_test[:, :, t],
                target_x_test[:, :, t],
                True,plot_flag)

            # =========================
            # Rectangular no-learning model
            # =========================
            dqr_rect.calibrate_crc(
                x_input_calib_per_t,
                target_x_calib[:, :, t],
                alpha
            )

            error_rect[t, epochs], IW_rect[t, epochs] = dqr_rect.inference_new(
                x_input_test_per_t,
                target_x_test[:, :, t],
                target_x_test[:, :, t],
                False,
                plot_flag
            )

            error_rect_before_cal[t, epochs], IW_rect_before_cal[t, epochs] = dqr_rect.inference_new(
                x_input_test_per_t,
                target_x_test[:, :, t],
                target_x_test[:, :, t],
                True,
                plot_flag
            )

            # =========================
            # Elliptical no-learning model
            # =========================
            dqr_elip.calibrate_ellip_crc(
                x_input_calib_per_t,
                target_x_calib[:, :, t],
            )

            error_elip_crc[t, epochs], IW_elip_crc[t, epochs] = dqr_elip.inference_new(
                x_input_test_per_t,
                target_x_test[:, :, t],
                target_x_test[:, :, t],
                False,
                plot_flag
            )

            error_elip_crc_before_cal[t, epochs], IW_elip_crc_before_cal[t, epochs] = dqr_elip.inference_new(
                x_input_test_per_t,
                target_x_test[:, :, t],
                target_x_test[:, :, t],
                True,
                plot_flag
            )

# ==========================================
# Average over epochs
# ==========================================
error_NPDQR = error_NPDQR.mean(dim=1)
IW_NPQR = IW_NPQR.mean(dim=1)

error_NPDQR_before_cal = error_NPDQR_before_cal.mean(dim=1)
IW_NPQR_before_cal = IW_NPQR_before_cal.mean(dim=1)

# NEW: small DQR
error_NPDQR_small = error_NPDQR_small.mean(dim=1)
IW_NPQR_small = IW_NPQR_small.mean(dim=1)

error_NPDQR_small_before_cal = error_NPDQR_small_before_cal.mean(dim=1)
IW_NPQR_small_before_cal = IW_NPQR_small_before_cal.mean(dim=1)

error_rect = error_rect.mean(dim=1)
IW_rect = IW_rect.mean(dim=1)

error_rect_before_cal = error_rect_before_cal.mean(dim=1)
IW_rect_before_cal = IW_rect_before_cal.mean(dim=1)

error_elip_crc = error_elip_crc.mean(dim=1)
IW_elip_crc = IW_elip_crc.mean(dim=1)

error_elip_crc_before_cal = error_elip_crc_before_cal.mean(dim=1)
IW_elip_crc_before_cal = IW_elip_crc_before_cal.mean(dim=1)

# ==========================================
# Convert to numpy
# ==========================================
err_dqr = torch.as_tensor(error_NPDQR).cpu().numpy()
iw_dqr = torch.as_tensor(IW_NPQR).cpu().numpy()

err_dqr_before = torch.as_tensor(error_NPDQR_before_cal).cpu().numpy()
iw_dqr_before = torch.as_tensor(IW_NPQR_before_cal).cpu().numpy()

# NEW: small DQR
err_dqr_small = torch.as_tensor(error_NPDQR_small).cpu().numpy()
iw_dqr_small = torch.as_tensor(IW_NPQR_small).cpu().numpy()

err_dqr_small_before = torch.as_tensor(error_NPDQR_small_before_cal).cpu().numpy()
iw_dqr_small_before = torch.as_tensor(IW_NPQR_small_before_cal).cpu().numpy()

err_rect = torch.as_tensor(error_rect).cpu().numpy()
iw_rect = torch.as_tensor(IW_rect).cpu().numpy()

err_rect_before = torch.as_tensor(error_rect_before_cal).cpu().numpy()
iw_rect_before = torch.as_tensor(IW_rect_before_cal).cpu().numpy()

err_elip = torch.as_tensor(error_elip_crc).cpu().numpy()
iw_elip = torch.as_tensor(IW_elip_crc).cpu().numpy()

err_elip_before = torch.as_tensor(error_elip_crc_before_cal).cpu().numpy()
iw_elip_before = torch.as_tensor(IW_elip_crc_before_cal).cpu().numpy()

# ==========================================
# Time axis
# ==========================================
T = len(err_dqr)
t_axis = np.arange(T)

# ==========================================
# FIGURE 1 : ERROR
# ==========================================
plt.figure(figsize=(12, 6))

# NP-DQR
plt.plot(
    t_axis,
    err_dqr,
    color="green",
    linewidth=2,
    label="NP-DQR"
)

plt.plot(
    t_axis,
    err_dqr_before,
    color="green",
    linestyle="--",
    linewidth=2,
    label="NP-DQR Before Calibration"
)

# NEW: NP-DQR small
plt.plot(
    t_axis,
    err_dqr_small,
    color="red",
    linewidth=2,
    label="NP-DQR Small"
)

plt.plot(
    t_axis,
    err_dqr_small_before,
    color="red",
    linestyle="--",
    linewidth=2,
    label="NP-DQR Small Before Calibration"
)

# Rectangular
plt.plot(
    t_axis,
    err_rect,
    color="orange",
    linewidth=2,
    label="Rectangular"
)

plt.plot(
    t_axis,
    err_rect_before,
    color="orange",
    linestyle="--",
    linewidth=2,
    label="Rectangular Before Calibration"
)

# Elliptical
plt.plot(
    t_axis,
    err_elip,
    color="blue",
    linewidth=2,
    label="Elliptical"
)

plt.plot(
    t_axis,
    err_elip_before,
    color="blue",
    linestyle="--",
    linewidth=2,
    label="Elliptical Before Calibration"
)

plt.xlabel("Time step t")
plt.ylabel("Error")
plt.title("Error vs Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("error_vs_time.png", dpi=200)
plt.show()
plt.close()

# ==========================================
# FIGURE 2 : COVERAGE AREA
# ==========================================
plt.figure(figsize=(12, 6))

# NP-DQR
plt.plot(
    t_axis,
    iw_dqr,
    color="green",
    linewidth=2,
    label="NP-DQR"
)

plt.plot(
    t_axis,
    iw_dqr_before,
    color="green",
    linestyle="--",
    linewidth=2,
    label="NP-DQR Before Calibration"
)

# NEW: NP-DQR small
plt.plot(
    t_axis,
    iw_dqr_small,
    color="red",
    linewidth=2,
    label="NP-DQR Small"
)

plt.plot(
    t_axis,
    iw_dqr_small_before,
    color="red",
    linestyle="--",
    linewidth=2,
    label="NP-DQR Small Before Calibration"
)

# Rectangular
plt.plot(
    t_axis,
    iw_rect,
    color="orange",
    linewidth=2,
    label="Rectangular"
)

plt.plot(
    t_axis,
    iw_rect_before,
    color="orange",
    linestyle="--",
    linewidth=2,
    label="Rectangular Before Calibration"
)

# Elliptical
plt.plot(
    t_axis,
    iw_elip,
    color="blue",
    linewidth=2,
    label="Elliptical"
)

plt.plot(
    t_axis,
    iw_elip_before,
    color="blue",
    linestyle="--",
    linewidth=2,
    label="Elliptical Before Calibration"
)

plt.xlabel("Time step t")
plt.ylabel("Coverage Area")
plt.title("Coverage Area vs Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("coverage_vs_time.png", dpi=200)
plt.show()
plt.close()

