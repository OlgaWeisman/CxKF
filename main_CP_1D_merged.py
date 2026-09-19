import sys
from datetime import datetime

import torch
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# Choose scenario
# =====================================================
scenario = "nonlinear_partial"
# Options:
# "linear"
# "linear_nonGaussian"
# "nonlinear"
# "nonlinear_partial"

T = 100
T_test = 100
alpha = 0.05
vdB = -20
r2 = torch.tensor([1.0])
v = 10 ** (vdB / 10)
q2 = torch.mul(v, r2)

# =====================================================
# Device
# =====================================================
torch.pi = torch.acos(torch.zeros(1)).item() * 2

if False:  # torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")
    print("Running on the CPU")

print("Pipeline Start")

# =====================================================
# Time
# =====================================================
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

path_results = "RTSNet/"

# =====================================================
# Scenario-specific imports and paths
# =====================================================
if scenario == "linear":
    from Linear_sysmdl import SystemModel
    from KalmanFilter_EstOnly_test import KFTest

    path_model = "Simulations/Linear_canonical/r=1_q=1"
    filter_type = "KF"
    model_kind = "linear"
    scenario_name = "LC-Gauss"

elif scenario == "linear_nonGaussian":
    from Linear_nonGaussian_sysmdl import SystemModel
    from KalmanFilter_EstOnly_test import KFTest

    path_model = "Simulations/Linear_canonical/r=1_q=1"
    filter_type = "KF"
    model_kind = "linear"
    scenario_name = "LC-non-Gauss"

elif scenario == "nonlinear":
    from Extended_sysmdl import SystemModel
    from EKF_EstOnly_test import EKFTest

    path_model = "Simulations/nonLinear_1D"
    filter_type = "EKF"
    model_kind = "nonlinear"
    scenario_name = "non-Linear"

elif scenario == "nonlinear_partial":
    from Extended_sysmdl import SystemModel
    from EKF_EstOnly_test import EKFTest

    path_model = "Simulations/nonLinear_1D"
    filter_type = "EKF"
    model_kind = "nonlinear_partial"
    scenario_name = "non-Linear-partial"

else:
    raise ValueError(f"Unknown scenario: {scenario}")

sys.path.insert(1, path_model)

from Extended_data_cp import (
    DataGen,
    DataLoader,
    DataLoader_GPU,
    Decimate_and_perturbate_Data,
    Short_Traj_Split,
    N_E,
    N_CV,
    N_T,
    J_T,
)

from CP_test import compute_gaussian_quantiles, clalibration_residual_quantiles

from model import f, h, f_tylor

if model_kind == "linear":
    from parameters import (
        m,
        n,
        m1x_0,
        m2x_0,
        Q,
        R,
        Q_mod,
        R_mod,
        sigma_r,
        sigma_q,
        F,
        H,
        F_mod,
        H_mod,
    )
else:
    from parameters import T as T_from_params, m1x_0, m2x_0, m, n


# =====================================================
# Helper functions
# =====================================================
def build_system_model(q, r):
    if model_kind == "linear":
        sys_model = SystemModel(F, q, H, r, T, T_test)
    else:
        try:
            sys_model = SystemModel(
                f,
                q,
                h,
                r,
                T,
                T_test,
                m,
                n,
                "1D",
                device=dev,
            )
            sys_model_partial = SystemModel(
                f_tylor,
                q,
                h,
                r,
                T,
                T_test,
                m,
                n,
                "1D",
            )
        except TypeError:
            sys_model = SystemModel(
                f,
                q,
                h,
                r,
                T,
                T_test,
                m,
                n,
                "1D",
            )
            sys_model_partial = SystemModel(
                f_tylor,
                q,
                h,
                r,
                T,
                T_test,
                m,
                n,
                "1D",
            )

    sys_model.InitSequence(m1x_0, m2x_0)
    sys_model_partial.InitSequence(m1x_0, m2x_0)
    return sys_model,sys_model_partial


def run_filter(sys_model, train_input, train_target):
    if filter_type == "KF":
        return KFTest(sys_model, train_input)

    if scenario == "nonlinear":
        return EKFTest(sys_model, train_input, train_target)

    if scenario == "nonlinear_partial":
        return EKFTest(sys_model, train_input, train_target)

    raise ValueError(f"Unsupported filter configuration: {scenario}")


def prepare_sigma_for_quantiles(sigma_array_train):
    # EKF sometimes returns covariance as [N, d, d, T].
    # For 1D CP quantiles we need [N, d, T].
    if sigma_array_train.ndim == 4:
        return torch.diagonal(
            sigma_array_train,
            dim1=1,
            dim2=2,
        ).permute(0, 2, 1)

    return sigma_array_train


# =====================================================
# Main experiment
# =====================================================


for index in range(len(r2)):
    print("========================================")
    print("Scenario:", scenario)
    print("1/r2 [dB]: ", 10 * torch.log10(1 / r2[index]))
    print("1/q2 [dB]: ", 10 * torch.log10(1 / q2[index]))

    # True model
    r = torch.sqrt(r2[index])
    q = torch.sqrt(q2[index])
    sys_model, sys_model_partial = build_system_model(q, r)

    # =====================================================
    # Data generation/loading
    # =====================================================
    dataFolderName = path_model + "/"

    if scenario == "linear_nonGaussian":
        dataFileName = [
            "1x1_rq-1010_T100_nonG.pt",
            "1x1_rq020_T100_nonG.pt",
            "1x1_rq3050_T100_nonG.pt",
        ]
    elif scenario == "linear":
        dataFileName = [
            "1x1_rq-1010_T100.pt",
            "1x1_rq020_T100.pt",
            "1x1_rq3050_T100.pt",
        ]
    else:
        dataFileName = [
            "1x1_rq-1010_T100.pt",
            "1x1_rq020_T100.pt",
            "1x1_rq1030_T100.pt",
            "1x1_rq2040_T100.pt",
            "1x1_rq3050_T100.pt",
        ]

    current_data_file = dataFolderName + dataFileName[index]

    # print("Start Data Gen")
    # DataGen(
    #     sys_model,
    #     current_data_file,
    #     T,
    #     T_test,
    #     randomInit=False,
    # )

    print("Data Load")
    [
        train_input,
        train_target,
        cv_input,
        cv_target,
        test_input,
        test_target,
    ] = DataLoader(current_data_file)

    print("trainset size:", train_target.size())
    print("cvset size:", cv_target.size())
    print("testset size:", test_target.size())

    # =====================================================
    # Filter evaluation
    # =====================================================
    print("Evaluate Filter")

    if scenario == "nonlinear_partial":
        x_array_train, sigma_array_train = run_filter(
            sys_model_partial,
            train_input,
            train_target,
        )
        x_array_cv, sigma_array_cv = run_filter(
            sys_model_partial,
            cv_input,
            cv_target,
        )
    else:
        x_array_train, sigma_array_train = run_filter(
            sys_model,
            train_input,
            train_target,
        )

        x_array_cv, sigma_array_cv = run_filter(
            sys_model,
            cv_input,
            cv_target,
        )
    sigma_for_quantiles_cv = prepare_sigma_for_quantiles(sigma_array_cv)
    if sigma_for_quantiles_cv.dim() < 3:
        sigma_for_quantiles_cv = sigma_for_quantiles_cv.unsqueeze(1)
        x_array_cv = x_array_cv.unsqueeze(1)


    sigma_for_quantiles = prepare_sigma_for_quantiles(sigma_array_train)

    # =====================================================
    # Gaussian quantiles
    # =====================================================
    [q_low, q_hi, q_half] = compute_gaussian_quantiles(
        x_array_train,
        sigma_for_quantiles,
        alpha,
    )

    [q_low_aT, q_hi_aT, q_half_aT] = compute_gaussian_quantiles(
        x_array_train,
        sigma_for_quantiles,
        alpha / T,
    )

    # Since all scenarios here are 1D, squeeze the state dimension.
    if x_array_train.ndim == 3:
        x_array_train = x_array_train.squeeze(1)


    if q_low.ndim == 3:
        q_low = q_low.squeeze(1)
        q_hi = q_hi.squeeze(1)
        q_half = q_half.squeeze(1)

    if q_low_aT.ndim == 3:
        q_low_aT = q_low_aT.squeeze(1)
        q_hi_aT = q_hi_aT.squeeze(1)
        q_half_aT = q_half_aT.squeeze(1)

    # =====================================================
    # Conformal calibration
    # =====================================================
    title_value = 10 * torch.log10(1 / r2[index]).item()
    # for test
    n_used = min(1000, x_array_train.shape[0])
    n_cal = int(np.ceil(0.8 * n_used))
    rho = 0.99
    weights = rho ** np.arange(n_cal, 0, -1)
    [c, w, s] = clalibration_residual_quantiles(
        train_input,
        x_array_train,
        sigma_for_quantiles,
        train_target,
        q_low,
        q_hi,
        J_T,
        alpha,
        title_value,
        q_half,
        q_low_aT,
        q_hi_aT,
        title_value,
        scenario_name,
        torch.cat([x_array_cv, sigma_for_quantiles_cv], dim=1),
        cv_target,
        cv_input,
    )

    print("Done scenario:", scenario)
