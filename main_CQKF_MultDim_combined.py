import os
import sys
import copy
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from DQREstimator import DQREstimator, rectangular_no_learning
from mahalanobis_elliptical_region_no_learning import MahalanobisEllipticalRegionNoLearning

# ======================================================
# User configuration
# ======================================================
scenario = "linear"   # options: "linear", "nonlinear", "lorenz"
model_type = "KF"       # options: "KF", "EKF", "UKF"

use_train_and_calib_for_no_learning = True

T = 10
T_test = 10
alpha = 0.05
Epochs_num = 5
train_model = True      # True: train and save models, False: load saved models
load_path = None

r2 = torch.tensor([20])   # for linear you may prefer torch.tensor([1])
vdB = -20                # q2/r2 in dB
use_cuda = False

# ======================================================
# Device
# ======================================================
torch.pi = torch.acos(torch.zeros(1)).item() * 2

if use_cuda and torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")
    print("Running on the CPU")

print("Pipeline Start")

# ======================================================
# Scenario-specific imports and settings
# ======================================================
if scenario == "linear":
    from Linear_sysmdl import SystemModel
    from Extended_data import DataGen, DataLoader
    from Extended_data import F, H, m1_0, m2_0, m, n
    from KalmanFilter_EstOnly_test import KFTest

    path_model = "Simulations/Linear_canonical/H=I/"
    data_folder = path_model
    data_files = [
        "1x1_rq-1010_T100_nonG.pt",
        "1x1_rq020_T100_nonG.pt",
        "1x1_rq3050_T100_nonG.pt",
    ]
    dim = m
    model_type = "KF"
    sys_model_name = "Linear"

elif scenario == "nonlinear":
    path_model = "Simulations/Toy_problems/"
    sys.path.insert(1, path_model)

    from Extended_sysmdl import SystemModel
    from Extended_data_cp import DataGen, DataLoader
    from EKF_EstOnly_test import EKFTest
    from UKF_test import UKFTest
    from model import f, h
    from parameters import m1x_0, m2x_0, m, n

    data_folder = path_model
    data_files = [
        "1x1_rq-1010_T100.pt",
        "1x1_rq020_T100.pt",
        "1x1_rq1030_T100.pt",
        "1x1_rq2040_T100.pt",
        "1x1_rq3050_T100.pt",
    ]
    dim = m
    sys_model_name = "Toy"

elif scenario == "lorenz":
    path_model = "Simulations/Lorenz_Atractor/"
    sys.path.insert(1, path_model)

    from Extended_sysmdl import SystemModel
    from Extended_data_cp import DataGen, DataLoader
    from EKF_EstOnly_test import EKFTest
    from model import f, h
    from parameters import m1x_0, m2x_0, m, n

    data_folder = path_model
    data_files = [
        "1x1_rq-1010_T100.pt",
        "1x1_rq020_T100.pt",
        "1x1_rq1030_T100.pt",
        "1x1_rq2040_T100.pt",
        "1x1_rq3050_T100.pt",
    ]
    dim = m
    model_type = "EKF"
    sys_model_name = "Lorenz_Atractor"

else:
    raise ValueError("scenario must be one of: 'linear', 'nonlinear', 'lorenz'")

os.makedirs(path_model, exist_ok=True)

# ======================================================
# Helpers
# ======================================================
def plot_inference_regions(save_dir="inference_containers"):

    files = sorted(
        [f for f in os.listdir(save_dir) if f.endswith(".pt")]
    )

    plt.figure(figsize=(8, 8))

    first = True

    for file in files:

        data = torch.load(os.path.join(save_dir, file), map_location="cpu")

        model_name = data["model_name"]
        z_grid = data["z_grid"].numpy()
        z_in_region = data["z_in_region"].numpy()
        x = data["x0"].squeeze().numpy()
        y = data["y0"].squeeze().numpy()

        # Plot grid only once
        if first:
            plt.scatter(
                z_grid[:, 0],
                z_grid[:, 1],
                s=5,
                color="lightgray",
                alpha=0.15,
                label="Grid",
            )
            first = False

        # Prediction region
        plt.scatter(
            z_in_region[:, 0],
            z_in_region[:, 1],
            s=10,
            alpha=0.12,
            label=model_name,
        )

    # Estimated point
    plt.scatter(
        x[0],
        x[1],
        s=180,
        marker="*",
        color="red",
        edgecolors="black",
        linewidths=1.2,
        label="Estimated",
        zorder=10,
    )

    # Target point
    plt.scatter(
        y[0],
        y[1],
        s=180,
        marker="*",
        color="limegreen",
        edgecolors="black",
        linewidths=1.2,
        label="Target",
        zorder=10,
    )

    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$", fontsize=14)
    plt.title("Prediction Regions", fontsize=16)
    plt.axis("equal")
    plt.grid(True)

    # Legend
    leg = plt.legend(markerscale=1.3, fontsize=12)

    # Make legend markers opaque
    for handle in leg.legend_handles:
        handle.set_alpha(1)

    plt.tight_layout()
    plt.show()
def make_rectangular_directions(d: int) -> torch.Tensor:
    """Create directions [e1, -e1, e2, -e2, ...] for rectangular no-learning model."""
    directions = []
    for i in range(d):
        e = torch.zeros(d)
        e[i] = 1.0
        directions.append(e)
        directions.append(-e)
    return torch.stack(directions, dim=0)


def make_safe_name(value) -> str:
    return str(value).replace(".", "p").replace(" ", "").replace("[", "").replace("]", "")


def get_filter_outputs(sys_model, train_input, train_target, cv_input, cv_target, test_input, test_target):
    """Run/load KF/EKF/UKF outputs."""
    file_name = (
        f"{model_type}_outputs"
        f"_{scenario}"
        f"_T{T}"
        f"_r2{make_safe_name(r2)}"
        f"_alpha{make_safe_name(alpha)}"
        ".pt"
    )
    output_path = os.path.join(path_model, file_name)

    if train_model:
        print(f"Computing {model_type} outputs...")

        if scenario == "linear":
            x_array_train, sigma_array_train = KFTest(sys_model, train_input)
            x_array_cv, sigma_array_cv = KFTest(sys_model, cv_input)
            x_array_test, sigma_array_test = KFTest(sys_model, test_input)

        elif model_type == "UKF":
            _, _, _, x_array_train, sigma_array_train = UKFTest(sys_model, train_input, train_target)
            _, _, _, x_array_cv, sigma_array_cv = UKFTest(sys_model, cv_input, cv_target)
            _, _, _, x_array_test, sigma_array_test = UKFTest(sys_model, test_input, test_target)

        elif model_type == "EKF":
            x_array_train, sigma_array_train = EKFTest(sys_model, train_input, train_target)
            x_array_cv, sigma_array_cv = EKFTest(sys_model, cv_input, cv_target)
            x_array_test, sigma_array_test = EKFTest(sys_model, test_input, test_target)

        else:
            raise ValueError("For nonlinear/lorenz, model_type must be 'EKF' or 'UKF'")

        torch.save(
            {
                "x_array_train": x_array_train,
                "sigma_array_train": sigma_array_train,
                "x_array_cv": x_array_cv,
                "sigma_array_cv": sigma_array_cv,
                "x_array_test": x_array_test,
                "sigma_array_test": sigma_array_test,
            },
            output_path,
        )
        print(f"Saved filter outputs to {output_path}")

    else:
        print(f"Loading {model_type} outputs from {output_path}...")
        data = torch.load(output_path, map_location=dev)
        x_array_train = data["x_array_train"]
        sigma_array_train = data["sigma_array_train"]
        x_array_cv = data["x_array_cv"]
        sigma_array_cv = data["sigma_array_cv"]
        x_array_test = data["x_array_test"]
        sigma_array_test = data["sigma_array_test"]

    return x_array_train, sigma_array_train, x_array_cv, sigma_array_cv, x_array_test, sigma_array_test


def save_dqr_model(dqr_obj, model_name: str):
    torch.save(
        {
            "model_state_dict": dqr_obj.model.state_dict(),
            "u_list": dqr_obj.u_list,
        },
        os.path.join(path_model, model_name),
    )


# ======================================================
# Time and W&B config
# ======================================================
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

v = 10 ** (vdB / 10)
q2 = torch.mul(v, r2)

# ======================================================
# Main loop over r2 values
# ======================================================
for index in range(len(r2)):
    print("1/r2 [dB]: ", 10 * torch.log10(1 / r2[index]))
    print("1/q2 [dB]: ", 10 * torch.log10(1 / q2[index]))

    r = torch.sqrt(r2[index])
    q = torch.sqrt(q2[index])

    # -----------------------------
    # Build system model
    # -----------------------------
    if scenario == "linear":
        sys_model = SystemModel(F, q, H, r, T, T_test)
        sys_model.InitSequence(m1_0, m2_0)
    else:
        sys_model = SystemModel(f, q, h, r, T, T_test, m, n, sys_model_name, device=dev)
        sys_model.InitSequence(m1x_0, m2x_0)

    # -----------------------------
    # Data generation/loading
    # -----------------------------
    data_file = os.path.join(data_folder, data_files[index])

    if train_model:
        print("Start Data Gen")
        DataGen(sys_model, data_file, T, T_test, randomInit=False)

    print("Data Load")
    train_input, train_target, cv_input, cv_target, test_input, test_target = DataLoader(data_file)

    print("trainset size:", train_target.size())
    print("cvset size:", cv_target.size())
    print("testset size:", test_target.size())

    # -----------------------------
    # KF/EKF/UKF outputs
    # -----------------------------
    x_array_train, sigma_array_train, x_array_cv, sigma_array_cv, x_array_test, sigma_array_test = get_filter_outputs(
        sys_model, train_input, train_target, cv_input, cv_target, test_input, test_target
    )
    N = x_array_test.shape[0]
    Test_samples = int(test_target.shape[0]*0.2)
    # -----------------------------
    # Storage
    # -----------------------------
    error_NPDQR = torch.empty(T, Epochs_num)
    IW_NPQR = torch.empty(T, Epochs_num)
    s_dqr = torch.empty(Test_samples, T, Epochs_num)
    error_NPDQR_before_cal = torch.empty(T, Epochs_num)
    IW_NPQR_before_cal = torch.empty(T, Epochs_num)
    s_dqr_before_cal = torch.empty(Test_samples, T, Epochs_num)

    error_NPDQR_small = torch.empty(T, Epochs_num)
    IW_NPQR_small = torch.empty(T, Epochs_num)
    s_dqr_small = torch.empty(Test_samples, T, Epochs_num)
    error_NPDQR_small_before_cal = torch.empty(T, Epochs_num)
    IW_NPQR_small_before_cal = torch.empty(T, Epochs_num)
    s_dqr_small_before_cal = torch.empty(Test_samples, T, Epochs_num)

    error_rect = torch.empty(T, Epochs_num)
    IW_rect = torch.empty(T, Epochs_num)
    s_rect = torch.empty(Test_samples, T, Epochs_num)
    error_rect_before_cal = torch.empty(T, Epochs_num)
    IW_rect_before_cal = torch.empty(T, Epochs_num)
    s_rect_before_cal = torch.empty(Test_samples, T, Epochs_num)

    error_elip_crc = torch.empty(T, Epochs_num)
    IW_elip_crc = torch.empty(T, Epochs_num)
    s_elip = torch.empty(Test_samples, T, Epochs_num)
    error_elip_crc_before_cal = torch.empty(T, Epochs_num)
    IW_elip_crc_before_cal = torch.empty(T, Epochs_num)
    s_elip_before_cal = torch.empty(Test_samples, T, Epochs_num)
    # -----------------------------
    # DQR inputs
    # -----------------------------

    x_input_train = torch.cat(
        [x_array_train, torch.sqrt(sigma_array_train.flatten(start_dim=1, end_dim=2))],
        dim=1,
    )
    x_input_cv = torch.cat(
        [x_array_cv, torch.sqrt(sigma_array_cv.flatten(start_dim=1, end_dim=2))],
        dim=1,
    )

    # -----------------------------
    # DQR models
    # -----------------------------
    dqr = DQREstimator(
        hs_str="[128,128]",
        num_ep=500,
        num_u=128,
        batch_size=128,
        lr=1e-3,
        wd=1e-4,
        dropout=0.01,
        device="cpu",
        patience=100,
        debug_plot=False,
    )

    dqr_fixing = DQREstimator(
        hs_str="[128,128]",
        num_ep=500,
        num_u=128,
        batch_size=128,
        lr=1e-3,
        wd=1e-4,
        dropout=0.01,
        device="cpu",
        patience=100,
        debug_plot=False,
    )

    dqr_small = DQREstimator(
        hs_str="[128,128]",
        num_ep=500,
        num_u=16,
        batch_size=128,
        lr=1e-3,
        wd=1e-4,
        dropout=0.01,
        device="cpu",
        patience=100,
        debug_plot=False,
    )

    # -----------------------------
    # No-learning elliptical model
    # -----------------------------
    dqr_elip = copy.deepcopy(dqr)
    dqr_elip.num_u = None
    dqr_elip.model = MahalanobisEllipticalRegionNoLearning(dim=dim, alpha=alpha)

    # -----------------------------
    # No-learning rectangular model
    # -----------------------------
    dqr_rect = copy.deepcopy(dqr)
    dqr_rect.u_list = make_rectangular_directions(dim)
    dqr_rect.model = lambda x_hat: rectangular_no_learning(x_hat=x_hat, alpha=alpha, d=dim)

    common_config = {
        "scenario": scenario,
        "model_type": model_type,
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

    group_name = f"{scenario}_{model_type}_compare_learning_{strTime}"

    # ======================================================
    # Train/load DQR
    # ======================================================
    model_name = f"dqr_model_{scenario}_{model_type}_T{T}_r2{make_safe_name(r2[index])}_alpha{alpha:.3f}.pt"
    current_load_path = os.path.join(path_model, model_name) if not train_model else None

    with wandb.init(
        project="Linear Gaussian",
        name=f"DQR_{scenario}_{model_type}_{strTime}",
        group=group_name,
        job_type="DQR",
        config={**common_config, "method": "DQR"},
        reinit="finish_previous",
    ):
        dqr.fit_nonLinear_forall_seq(
            x_input_train,
            train_target,
            x_input_cv,
            cv_target,
            tau=alpha / 7,
            log_wandb=True,
            load_path=current_load_path,
        )
        if train_model:
            save_dqr_model(dqr, model_name)

    # ======================================================
    # Train/load DQR Gaussian correction
    # ======================================================
    model_name = f"dqr_fixing_model_{scenario}_{model_type}_T{T}_r2{make_safe_name(r2[index])}_alpha{alpha:.3f}.pt"
    current_load_path = os.path.join(path_model, model_name) if not train_model else None

    with wandb.init(
        project="Linear Gaussian",
        name=f"DQR_fixing_{scenario}_{model_type}_{strTime}",
        group=group_name,
        job_type="Gaussian_correction",
        config={**common_config, "method": "Gaussian_correction"},
        reinit="finish_previous",
    ):
        dqr_fixing.fit_gaussian_correction_forall_seq(
            x_input_train,
            train_target,
            x_input_cv,
            cv_target,
            tau=alpha / 7,
            log_wandb=True,
            load_path=current_load_path,
        )
        if train_model:
            save_dqr_model(dqr_fixing, model_name)

    wandb.finish()

    # ======================================================
    # Train/load small DQR
    # ======================================================
    model_name = f"dqr_small_model_{scenario}_{model_type}_T{T}_r2{make_safe_name(r2[index])}_alpha{alpha:.3f}.pt"
    current_load_path = os.path.join(path_model, model_name) if not train_model else None

    dqr_small.fit_nonLinear_forall_seq(
        x_input_train,
        train_target,
        x_input_cv,
        cv_target,
        tau=alpha / 7,
        log_wandb=False,
        load_path=current_load_path,
    )
    if train_model:
        save_dqr_model(dqr_small, model_name)

    # ======================================================
    # Calibration/testing loop
    # ======================================================
    for epochs in range(Epochs_num):
        indices = np.random.permutation(N)
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

        for t in range(T):
            x_input_calib_per_t = torch.cat(
                [hat_x_calib[:, :, t], torch.sqrt(sigma_calib_flat[:, :, t])],
                dim=1,
            )
            x_input_test_per_t = torch.cat(
                [hat_x_test[:, :, t], torch.sqrt(sigma_test_flat[:, :, t])],
                dim=1,
            )

            # DQR
            dqr.calibrate_crc(x_input_calib_per_t, target_x_calib[:, :, t], alpha)
            error_NPDQR[t, epochs], IW_NPQR[t, epochs],s_dqr[:, t, epochs] = dqr.inference_new(
                x_input_test_per_t, target_x_test[:, :, t], target_x_test[:, :, t], False
            )
            error_NPDQR_before_cal[t, epochs], IW_NPQR_before_cal[t, epochs], s_dqr_before_cal[:, t, epochs] = dqr.inference_new(
                x_input_test_per_t, target_x_test[:, :, t], target_x_test[:, :, t], True
            )

            # Small DQR
            dqr_small.calibrate_crc(x_input_calib_per_t, target_x_calib[:, :, t], alpha)
            error_NPDQR_small[t, epochs], IW_NPQR_small[t, epochs], s_dqr_small[:, t, epochs]  = dqr_small.inference_new(
                x_input_test_per_t, target_x_test[:, :, t], target_x_test[:, :, t], False
            )
            error_NPDQR_small_before_cal[t, epochs], IW_NPQR_small_before_cal[t, epochs], s_dqr_small_before_cal[:, t, epochs] = dqr_small.inference_new(
                x_input_test_per_t, target_x_test[:, :, t], target_x_test[:, :, t], True
            )
            if use_train_and_calib_for_no_learning:

                x_input_rect_elip = torch.cat(
                    [x_input_train[:, :, t], x_input_calib_per_t],
                    dim=0
                )

                target_rect_elip = torch.cat(
                    [train_target[:, :, t], target_x_calib[:, :, t]],
                    dim=0
                )

            else:

                x_input_rect_elip = x_input_calib_per_t
                target_rect_elip = target_x_calib[:, :, t]

            # Rectangular no-learning
            dqr_rect.calibrate_crc(
                x_input_rect_elip,
                target_rect_elip,
                alpha
            )
            error_rect[t, epochs], IW_rect[t, epochs], s_rect[:, t, epochs] = dqr_rect.inference_new(
                x_input_test_per_t, target_x_test[:, :, t], target_x_test[:, :, t], False
            )
            error_rect_before_cal[t, epochs], IW_rect_before_cal[t, epochs], s_rect_before_cal[:, t, epochs] = dqr_rect.inference_new(
                x_input_test_per_t, target_x_test[:, :, t], target_x_test[:, :, t], True
            )

            # Elliptical no-learning
            dqr_elip.calibrate_ellip_crc(
                x_input_rect_elip,
                target_rect_elip
            )
            error_elip_crc[t, epochs], IW_elip_crc[t, epochs], s_elip[:, t, epochs] = dqr_elip.inference_new(
                x_input_test_per_t, target_x_test[:, :, t], target_x_test[:, :, t], False
            )
            error_elip_crc_before_cal[t, epochs], IW_elip_crc_before_cal[t, epochs], s_elip_before_cal[:, t, epochs] = dqr_elip.inference_new(
                x_input_test_per_t, target_x_test[:, :, t], target_x_test[:, :, t], True
            )
            # if epochs == 0:
            #     plot_inference_regions("inference_containers")

    # ======================================================
    # Average over epochs
    # ======================================================

    error_NPDQR = error_NPDQR.mean(dim=1)
    IW_NPQR = IW_NPQR.mean(dim=1)
    Traj_failure_dqr = s_dqr.sum(dim=1).mean()
    error_NPDQR_before_cal = error_NPDQR_before_cal.mean(dim=1)
    IW_NPQR_before_cal = IW_NPQR_before_cal.mean(dim=1)
    Traj_failure_dqr_before_cal = s_dqr_before_cal.sum(dim=1).mean()
    error_NPDQR_small = error_NPDQR_small.mean(dim=1)
    IW_NPQR_small = IW_NPQR_small.mean(dim=1)
    Traj_failure_dqr_small = s_dqr_small.sum(dim=1).mean()
    error_NPDQR_small_before_cal = error_NPDQR_small_before_cal.mean(dim=1)
    IW_NPQR_small_before_cal = IW_NPQR_small_before_cal.mean(dim=1)
    Traj_failure_dqr_small_before_cal = s_dqr_small_before_cal.sum(dim=1).mean()
    error_rect = error_rect.mean(dim=1)
    IW_rect = IW_rect.mean(dim=1)
    Traj_failure_rect = s_rect.sum(dim=1).mean()
    error_rect_before_cal = error_rect_before_cal.mean(dim=1)
    IW_rect_before_cal = IW_rect_before_cal.mean(dim=1)
    Traj_failure_rect_before_cal = s_rect_before_cal.sum(dim=1).mean()
    error_elip_crc = error_elip_crc.mean(dim=1)
    IW_elip_crc = IW_elip_crc.mean(dim=1)
    Traj_failure_elip = s_elip.sum(dim=1).mean()
    error_elip_crc_before_cal = error_elip_crc_before_cal.mean(dim=1)
    IW_elip_crc_before_cal = IW_elip_crc_before_cal.mean(dim=1)
    Traj_failure_elip_before_cal = s_elip_before_cal.sum(dim=1).mean()
    # ======================================================
    # Convert to numpy
    # ======================================================
    err_dqr = torch.as_tensor(error_NPDQR).cpu().numpy()
    iw_dqr = torch.as_tensor(IW_NPQR).cpu().numpy()
    traj_dqr = torch.as_tensor(Traj_failure_dqr).cpu().item()
    err_dqr_before = torch.as_tensor(error_NPDQR_before_cal).cpu().numpy()
    iw_dqr_before = torch.as_tensor(IW_NPQR_before_cal).cpu().numpy()
    traj_dqr_before = torch.as_tensor(Traj_failure_dqr_before_cal).cpu().item()
    err_dqr_small = torch.as_tensor(error_NPDQR_small).cpu().numpy()
    iw_dqr_small = torch.as_tensor(IW_NPQR_small).cpu().numpy()
    traj_dqr_small = torch.as_tensor(Traj_failure_dqr_small).cpu().item()
    err_dqr_small_before = torch.as_tensor(error_NPDQR_small_before_cal).cpu().numpy()
    iw_dqr_small_before = torch.as_tensor(IW_NPQR_small_before_cal).cpu().numpy()
    traj_dqr_small_before = torch.as_tensor(Traj_failure_dqr_small_before_cal).cpu().item()
    err_rect = torch.as_tensor(error_rect).cpu().numpy()
    iw_rect = torch.as_tensor(IW_rect).cpu().numpy()
    traj_rect = torch.as_tensor(Traj_failure_rect).cpu().item()
    err_rect_before = torch.as_tensor(error_rect_before_cal).cpu().numpy()
    iw_rect_before = torch.as_tensor(IW_rect_before_cal).cpu().numpy()
    traj_rect_before = torch.as_tensor(Traj_failure_rect_before_cal).cpu().item()
    err_elip = torch.as_tensor(error_elip_crc).cpu().numpy()
    iw_elip = torch.as_tensor(IW_elip_crc).cpu().numpy()
    traj_elip = torch.as_tensor(Traj_failure_elip).cpu().item()
    err_elip_before = torch.as_tensor(error_elip_crc_before_cal).cpu().numpy()
    iw_elip_before = torch.as_tensor(IW_elip_crc_before_cal).cpu().numpy()
    traj_elip_before = torch.as_tensor(Traj_failure_elip_before_cal).cpu().item()
    t_axis = np.arange(len(err_dqr))
    # ======================================================
    # Plot error
    # ======================================================
    plt.figure(figsize=(12, 6))
    plt.plot(t_axis, err_dqr, color="green", linewidth=2, label="NP-DQR")
    plt.plot(t_axis, err_dqr_before, color="green", linestyle="--", linewidth=2, label="NP-DQR Before Calibration")
    plt.plot(t_axis, err_dqr_small, color="red", linewidth=2, label="NP-DQR Small")
    plt.plot(t_axis, err_dqr_small_before, color="red", linestyle="--", linewidth=2, label="NP-DQR Small Before Calibration")
    plt.plot(t_axis, err_rect, color="orange", linewidth=2, label="Rectangular")
    plt.plot(t_axis, err_rect_before, color="orange", linestyle="--", linewidth=2, label="Rectangular Before Calibration")
    plt.plot(t_axis, err_elip, color="blue", linewidth=2, label="Elliptical")
    plt.plot(t_axis, err_elip_before, color="blue", linestyle="--", linewidth=2, label="Elliptical Before Calibration")
    plt.xlabel("Time step t")
    plt.ylabel("Error")
    plt.title(f"Error vs Time - {scenario} - {model_type}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"error_vs_time_{scenario}_{model_type}.png", dpi=200)
    plt.show()
    plt.close()
    # ======================================================
    # Plot coverage area
    # ======================================================
    plt.figure(figsize=(12, 6))
    plt.plot(
        t_axis, iw_dqr,
        color="green", linewidth=2,
        label=f"NP-DQR | Traj failure={traj_dqr:.4f}"
    )
    plt.plot(
        t_axis, iw_dqr_before,
        color="green", linestyle="--", linewidth=2,
        label=f"NP-DQR Before Cal | Traj failure={traj_dqr_before:.4f}"
    )
    plt.plot(
        t_axis, iw_dqr_small,
        color="red", linewidth=2,
        label=f"NP-DQR Small | Traj failure={traj_dqr_small:.4f}"
    )
    plt.plot(
        t_axis, iw_dqr_small_before,
        color="red", linestyle="--", linewidth=2,
        label=f"NP-DQR Small Before Cal | Traj failure={traj_dqr_small_before:.4f}"
    )
    plt.plot(
        t_axis, iw_rect,
        color="orange", linewidth=2,
        label=f"Rectangular | Traj failure={traj_rect:.4f}"
    )
    plt.plot(
        t_axis, iw_rect_before,
        color="orange", linestyle="--", linewidth=2,
        label=f"Rectangular Before Cal | Traj failure={traj_rect_before:.4f}"
    )
    plt.plot(
        t_axis, iw_elip,
        color="blue", linewidth=2,
        label=f"Elliptical | Traj failure={traj_elip:.4f}"
    )
    plt.plot(
        t_axis, iw_elip_before,
        color="blue", linestyle="--", linewidth=2,
        label=f"Elliptical Before Cal | Traj failure={traj_elip_before:.4f}"
    )
    plt.xlabel("Time step t")
    plt.ylabel("Coverage Area")
    plt.title(f"Coverage Area vs Time - {scenario} - {model_type}")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"coverage_vs_time_{scenario}_{model_type}.png", dpi=200)
    plt.show()
    plt.close()
