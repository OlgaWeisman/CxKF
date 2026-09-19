import torch
import math
from pathlib import Path
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import sys
import csv
import os
import re
# folder of current script (or project root if running from there)
project_dir = Path(__file__).resolve().parent

# go one level up → PythonCode
base_dir = project_dir.parent

# now append your target subfolder
target_path = base_dir / "timeParamCPScores-master" / "timeParamCPScores-master" / "code"

sys.path.append(str(target_path))

from gurobipyTutorial import optimzeTimeAlphasKKTNoMaxLowerBound, optimzeTimeAlphasKKT, optimzeTimeAlphasKKTNoMaxLowerBoundMinArea
from CQR_training import AllQuantileRegressor
from DR_CP_test import DRCPMDN
# from DistSplit import DistSplit
def compute_gaussian_quantiles(x_array_train, sigma_array_train,alpha):
    mu_hat = x_array_train
    sigma_hat = np.sqrt(sigma_array_train)

    # Compute x such that P(X <= x) = alpha
    hat_q_low = norm.ppf(alpha/2, loc=mu_hat, scale=sigma_hat)
    hat_q_high = norm.ppf(1-alpha/2, loc=mu_hat, scale=sigma_hat)
    hat_q_1_2 = norm.ppf(0.5, loc=mu_hat, scale=sigma_hat)
    return hat_q_low, hat_q_high, hat_q_1_2


def metric_calculation(curr_test_target, intervals):
    c = (curr_test_target[:, 0, :].numpy() < intervals[:, :, 0]) | (
                curr_test_target[:, 0, :].numpy() > intervals[:, :, 1])
    s = np.sum(c, -1) > 0
    w = intervals
    return c, s, w

def clalibration_residual_quantiles(train_input_long,train_KF_output_long, train_Sigma_KF_output_long,train_target_long ,q_low,q_hi, J_T, alpha, title_value, q_half,q_low_aT,q_hi_aT,snr,scenario,train_cqkf_input,train_cqr_target,train_cqr_input):
    plot_flag = True
    # List of algorithms
    algorithms = ["CQKF_trained","CQKF_TWj","CQR_Twj","CQR_trained"]#["CQKF_TWj","CQR_Twj","CQR_trained","CQKF_trained","CQR", "Gaussian only", "CQR_max", "CQKF-Bonf-sw","Gaussian only long", "time-series non Union", "Union-TS"]#["CQR", "Gaussian only"]##["CQR", "Gaussian only", "CQR_max", "CQR_alpha", "time-series non Union", "CQR_r", "dist-split"]
    #loop on trails
    train_KF_output = train_KF_output_long[:1000, :]
    train_target = train_target_long[:1000, :]
    train_Sigma_KF_output = train_Sigma_KF_output_long[:1000, :]
    train_input = train_input_long[:1000,:]
    T = train_KF_output.shape[1]
    # 20% testing 80% calibration
    num_of_calib = int(np.ceil(train_KF_output.size(0) * 0.8))
    num_of_tests = train_KF_output.size(0) - num_of_calib

    num_of_calib_long = int(np.ceil(train_KF_output_long.size(0) * 0.8))
    num_of_tests_long = train_KF_output_long.size(0) - num_of_calib_long

    #Initialization
    c = np.empty([J_T, num_of_tests, T])
    s = np.empty([J_T, num_of_tests])
    w = np.empty([J_T, num_of_tests, T, 2])

    c_trained = np.empty([J_T, num_of_tests, T])
    s_trained = np.empty([J_T, num_of_tests])
    w_trained = np.empty([J_T, num_of_tests, T, 2])

    c_cqr_tw = np.empty([J_T, num_of_tests, T])
    s_cqr_tw = np.empty([J_T, num_of_tests])
    w_cqr_tw = np.empty([J_T, num_of_tests, T, 2])

    c_cqkf_trained = np.empty([J_T, num_of_tests, T])
    s_cqkf_trained = np.empty([J_T, num_of_tests])
    w_cqkf_trained = np.empty([J_T, num_of_tests, T, 2])

    c_cqkf_tw = np.empty([J_T, num_of_tests, T])
    s_cqkf_tw = np.empty([J_T, num_of_tests])
    w_cqkf_tw = np.empty([J_T, num_of_tests, T, 2])

    c_long = np.empty([J_T, num_of_tests_long, T])
    s_long = np.empty([J_T, num_of_tests_long])
    w_long = np.empty([J_T, num_of_tests_long, T, 2])

    c_ts = np.empty([J_T, num_of_tests, T])
    s_ts = np.empty([J_T, num_of_tests])
    w_ts = np.empty([J_T, num_of_tests, T, 2])

    c_ts_alpha = np.empty([J_T, num_of_tests, T])
    s_ts_alpha = np.empty([J_T, num_of_tests])
    w_ts_alpha = np.empty([J_T, num_of_tests, T, 2])

    c_ts_only = np.empty([J_T, num_of_tests, T])
    s_ts_only = np.empty([J_T, num_of_tests])
    w_ts_only = np.empty([J_T, num_of_tests, T, 2])

    c_ts_only_union = np.empty([J_T, num_of_tests, T])
    s_ts_only_union = np.empty([J_T, num_of_tests])
    w_ts_only_union = np.empty([J_T, num_of_tests, T, 2])

    c_gu = np.empty([J_T, num_of_tests, T])
    s_gu = np.empty([J_T, num_of_tests])
    w_gu = np.empty([J_T, num_of_tests, T, 2])

    c_gu_long = np.empty([J_T, num_of_tests_long, T])
    s_gu_long = np.empty([J_T, num_of_tests_long])
    w_gu_long = np.empty([J_T, num_of_tests_long, T, 2])

    c_pav = np.empty([J_T, num_of_tests, T])
    s_pav = np.empty([J_T, num_of_tests])
    w_pav = np.empty([J_T, num_of_tests, T, 2])

    c_dist_split = np.empty([J_T, num_of_tests, T])
    s_dist_split = np.empty([J_T, num_of_tests])
    w_dist_split = np.empty([J_T, num_of_tests, T, 2])
    # for non-Union case alpha
    # Estimate alphas
    if "cqr + time-series non Union" in algorithms:
        shuffled_idx_all = torch.randperm(train_KF_output.size(0))
        shuffled_idx_alpha = shuffled_idx_all[:100]
        curr_target_alpha = train_target.index_select(0, shuffled_idx_alpha)
        curr_output_alpha = train_KF_output.index_select(0, shuffled_idx_alpha)
        curr_q_low_alpha = q_low[shuffled_idx_alpha]
        curr_q_hi_alpha = q_hi[shuffled_idx_alpha]

        error_low_a = curr_q_low_alpha - curr_target_alpha.squeeze().numpy()
        error_high_a = curr_target_alpha.squeeze().numpy() - curr_q_hi_alpha
        err_for_alpha = np.abs(np.maximum(error_high_a, error_low_a))
        R_vals = err_for_alpha.tolist()
        m = optimzeTimeAlphasKKTNoMaxLowerBound(R_vals, alpha, 100000)
        alphas = []
        for v in m.getVars():
            if "alphas" in v.varName:
                alphas.append(v.x)
            if "q" in v.varName:
                # print(v.x)
                print("obj: " + str(v.x))
    if "time-series non Union" in algorithms:
        shuffled_idx_all = torch.randperm(train_KF_output.size(0))
        shuffled_idx_alpha = shuffled_idx_all[:100]
        curr_target_alpha = train_target.index_select(0, shuffled_idx_alpha)
        curr_output_alpha = train_KF_output.index_select(0, shuffled_idx_alpha)
        # alphas calibration
        test_target_list = curr_target_alpha.squeeze().tolist()
        x_array_test_list = curr_output_alpha.tolist()
        R_vals = [[math.sqrt((test_target_list[i][j] - x_array_test_list[i][j]) ** 2 ) for j in range(len(test_target_list[i]))] for i in range(len(curr_target_alpha))]

        m = optimzeTimeAlphasKKTNoMaxLowerBound(R_vals, alpha, 100000)
        alphas_v1 = []
        for v in m.getVars():
            if "alphas" in v.varName:
                alphas_v1.append(v.x)
            if "q" in v.varName:
                # print(v.x)
                print("obj: " + str(v.x))

    if "CQKF_trained" in algorithms:


        quantiles = [alpha / 2, 1 - alpha / 2]
        shuffled_idx_all = torch.randperm(train_KF_output.size(0))
        # Train on calibration/training trajectories
        x_train_cqr = train_cqkf_input#.numpy().reshape(-1, 1)
        y_train_cqr = train_cqr_target#.numpy().reshape(-1)

        cqr_model = AllQuantileRegressor(
            quantiles=quantiles,
            in_shape=x_train_cqr.shape[1],
            hidden_size=64,
            dropout=0.1,
            lr=1e-3,
            target_coverage=1 - alpha,
        )

        cqr_model.fit(
            x_train_cqr,
            y_train_cqr,
            epochs=1000,
            batch_size=128,
            verbose=True,
        )
        if train_Sigma_KF_output.dim() < 3:
            train_Sigma_KF_output = train_Sigma_KF_output.unsqueeze(1)
        # Predict quantiles for all trajectories
        _, pred_quantiles = cqr_model.predict(
            torch.cat([train_KF_output.unsqueeze(1),train_Sigma_KF_output], dim=1)
        )

        q_low_trained = pred_quantiles[:, 0, :]

        q_hi_trained = pred_quantiles[:, 1, : ]
    if "CQR_trained" in algorithms:


        quantiles = [alpha / 2, 1 - alpha / 2]

        # Train on calibration/training trajectories
        x_train_cqr = train_cqr_input#.numpy().reshape(-1, 1)
        y_train_cqr = train_cqr_target#.numpy().reshape(-1)

        cqr_model = AllQuantileRegressor(
            quantiles=quantiles,
            in_shape=x_train_cqr.shape[1],
            hidden_size=64,
            dropout=0.1,
            lr=1e-3,
            target_coverage=1 - alpha,
        )

        cqr_model.fit(
            x_train_cqr,
            y_train_cqr,
            epochs=1000,
            batch_size=128,
            verbose=True,
        )

        _, pred_quantiles = cqr_model.predict(
            train_input
        )

        q_obs_low_trained = pred_quantiles[:, 0, :]

        q_obs_hi_trained = pred_quantiles[:, 1, :]


    for j in range(J_T):
        shuffled_idx_long = torch.randperm(train_KF_output_long.size(0))
        curr_output_long = train_KF_output_long.index_select(0, shuffled_idx_long)
        curr_target_long = train_target_long.index_select(0, shuffled_idx_long)
        # for long
        curr_q_low_long = q_low_aT[shuffled_idx_long]
        curr_q_hi_long = q_hi_aT[shuffled_idx_long]

        curr_train_output_long = curr_output_long[:num_of_calib_long]
        curr_train_target_long = curr_target_long[:num_of_calib_long]
        curr_train_q_low_long = curr_q_low_long[:num_of_calib_long]
        curr_train_q_hi_long = curr_q_hi_long[:num_of_calib_long]

        curr_test_output_long = curr_output_long[num_of_calib_long:]
        curr_test_target_long = curr_target_long[num_of_calib_long:]
        curr_test_q_low_long = curr_q_low_long[num_of_calib_long:]
        curr_test_q_hi_long = curr_q_hi_long[num_of_calib_long:]

        # for short
        shuffled_idx = torch.randperm(train_KF_output.size(0))
        curr_output = train_KF_output.index_select(0, shuffled_idx)
        curr_target = train_target.index_select(0, shuffled_idx)
        curr_q_low = q_low[shuffled_idx]
        curr_q_hi = q_hi[shuffled_idx]
        curr_q_half = q_half[shuffled_idx]


        curr_train_output = curr_output[:num_of_calib]
        curr_train_target = curr_target[:num_of_calib]
        curr_train_q_low = curr_q_low[:num_of_calib]
        curr_train_q_hi = curr_q_hi[:num_of_calib]
        curr_train_q_half = curr_q_half[:num_of_calib]


        curr_train_q_low_trained = q_low_trained[:num_of_calib]
        curr_train_q_hi_trained = q_hi_trained[:num_of_calib]

        curr_obs_train_q_low_trained = q_obs_low_trained[:num_of_calib]
        curr_obs_train_q_hi_trained = q_obs_hi_trained[:num_of_calib]

        curr_test_output = curr_output[num_of_calib:]
        curr_test_target = curr_target[num_of_calib:]
        curr_test_q_low = curr_q_low[num_of_calib:]
        curr_test_q_hi = curr_q_hi[num_of_calib:]
        curr_test_q_half = curr_q_half[num_of_calib:]

        curr_test_q_low_trained = q_low_trained[num_of_calib:]
        curr_test_q_hi_trained = q_hi_trained[num_of_calib:]

        curr_obs_test_q_low_trained = q_obs_low_trained[num_of_calib:]
        curr_obs_test_q_hi_trained = q_obs_hi_trained[num_of_calib:]

        # For general calculation
        error_low = curr_train_q_low - curr_train_target.squeeze().numpy()
        error_high = curr_train_target.squeeze().numpy() - curr_train_q_hi
        err = np.maximum(error_high, error_low)
        cal_scores = {0: np.sort(err, 0)[::-1]}
        nc = np.sort(cal_scores[0], 0)

        index = int(np.ceil((1 - alpha) * (nc.shape[0] + 1))) - 1
        index = min(max(index, 0), nc.shape[0] - 1)

        # for trained CQKF
        error_low_trained = curr_train_q_low_trained - curr_train_target.squeeze().numpy()
        error_high_trained = curr_train_target.squeeze().numpy() - curr_train_q_hi_trained
        err_trained = np.maximum(error_high_trained, error_low_trained)
        cal_scores_trained = {0: np.sort(err_trained, 0)[::-1]}
        nc_trained = np.sort(cal_scores_trained[0], 0)
        # for trained CQR
        error_low_trained = curr_obs_train_q_low_trained - curr_train_target.squeeze().numpy()
        error_high_trained = curr_train_target.squeeze().numpy() - curr_obs_train_q_hi_trained
        err_obs_trained = np.maximum(error_high_trained, error_low_trained)
        cal_scores_trained = {0: np.sort(err_obs_trained, 0)[::-1]}
        nc_obs_trained = np.sort(cal_scores_trained[0], 0)

        # For long general calculation
        error_low_long = curr_train_q_low_long - curr_train_target_long.squeeze().numpy()
        error_high_long = curr_train_target_long.squeeze().numpy() - curr_train_q_hi_long
        err_long = np.maximum(error_high_long, error_low_long)
        cal_scores_long = {0: np.sort(err_long, 0)[::-1]}
        nc_long = np.sort(cal_scores_long[0], 0)

        index_long = int(np.ceil((1 - alpha/T) * (nc_long.shape[0] + 1))) - 1
        index_long = min(max(index_long, 0), nc_long.shape[0] - 1)
        # CQR
        if "CQKF_trained" in algorithms:

            test_err = np.vstack([nc_trained[index, :], nc_trained[index, :]])
            intervals_cqkf_trained = np.zeros((num_of_tests, T, 2))
            intervals_cqkf_trained[:, :, 0] = curr_test_q_low_trained - np.tile(test_err[0, :], (num_of_tests, 1))
            intervals_cqkf_trained[:, :, 1] = curr_test_q_hi_trained + np.tile(test_err[1, :], (num_of_tests, 1))
            # Calculate metric result
            [c_cqkf_trained[j], s_cqkf_trained[j], w_cqkf_trained[j]] = metric_calculation(curr_test_target, intervals_cqkf_trained)
        if "CQKF_TWj" in algorithms:
            # time seq Union Bound
            alpha_err_trained = np.max(err_trained , axis=1)
            cal_scores = {0: np.sort(alpha_err_trained, 0)[::-1]}
            nc_twj = np.sort(cal_scores[0], 0)
            test_err = np.vstack([nc_twj[index], nc_twj[index]])
            intervals_cqkf_twj = np.zeros((num_of_tests, T, 2))
            intervals_cqkf_twj[:, :, 0] = curr_test_q_low_trained - np.tile(test_err[0, :], (num_of_tests, T))
            intervals_cqkf_twj[:, :, 1] = curr_test_q_hi_trained + np.tile(test_err[1, :], (num_of_tests, T))

            # Calculate metric result
            [c_cqkf_tw[j], s_cqkf_tw[j], w_cqkf_tw[j]] = metric_calculation(curr_test_target, intervals_cqkf_twj)

        if "CQR_trained" in algorithms:

            test_err = np.vstack([nc_obs_trained[index, :], nc_obs_trained[index, :]])
            intervals_cqr_trained = np.zeros((num_of_tests, T, 2))
            intervals_cqr_trained[:, :, 0] = curr_obs_test_q_low_trained - np.tile(test_err[0, :], (num_of_tests, 1))
            intervals_cqr_trained[:, :, 1] = curr_obs_test_q_hi_trained + np.tile(test_err[1, :], (num_of_tests, 1))
            # Calculate metric result
            [c_trained[j], s_trained[j], w_trained[j]] = metric_calculation(curr_test_target, intervals_cqr_trained)
        if "CQR_Twj" in algorithms:
            # time seq Union Bound
            alpha_err_trained = np.max(err_obs_trained , axis=1)
            cal_scores = {0: np.sort(alpha_err_trained, 0)[::-1]}
            nc_twj = np.sort(cal_scores[0], 0)
            test_err = np.vstack([nc_twj[index], nc_twj[index]])
            intervals_cqr_twj = np.zeros((num_of_tests, T, 2))
            intervals_cqr_twj[:, :, 0] = curr_obs_test_q_low_trained - np.tile(test_err[0, :], (num_of_tests, T))
            intervals_cqr_twj[:, :, 1] = curr_obs_test_q_hi_trained + np.tile(test_err[1, :], (num_of_tests, T))

            # Calculate metric result
            [c_cqr_tw[j], s_cqr_tw[j], w_cqr_tw[j]] = metric_calculation(curr_test_target, intervals_cqr_twj)
        if "CQR" in algorithms:
            test_err = np.vstack([nc[index, :], nc[index, :]])
            intervals = np.zeros((num_of_tests, T, 2))
            intervals[:, :, 0] = curr_test_q_low - np.tile(test_err[0, :], (num_of_tests, 1))
            intervals[:, :, 1] = curr_test_q_hi + np.tile(test_err[1, :], (num_of_tests, 1))
            # Calculate metric result
            [c[j], s[j], w[j]] = metric_calculation(curr_test_target, intervals)
        if "CQKF-Bonf-sw" in algorithms:
            test_err_long = np.vstack([nc_long[index_long, :], nc_long[index_long, :]])
            intervals_long = np.zeros((num_of_tests_long, T, 2))
            intervals_long[:, :, 0] = curr_test_q_low_long - np.tile(test_err_long[0, :], (num_of_tests_long, 1))
            intervals_long[:, :, 1] = curr_test_q_hi_long + np.tile(test_err_long[1, :], (num_of_tests_long, 1))
            # Calculate metric result
            [c_long[j], s_long[j], w_long[j]] = metric_calculation(curr_test_target_long, intervals_long)

        if "Gaussian only" in algorithms:
            intervals_gu = np.zeros((num_of_tests, T, 2))
            intervals_gu[:, :, 0] = curr_test_q_low
            intervals_gu[:, :, 1] = curr_test_q_hi
            # Calculate metric result
            [c_gu[j], s_gu[j], w_gu[j]] = metric_calculation(curr_test_target, intervals_gu)
        if "Gaussian only long" in algorithms:
            intervals_gu_long = np.zeros((num_of_tests_long, T, 2))
            intervals_gu_long[:, :, 0] = curr_test_q_low_long
            intervals_gu_long[:, :, 1] = curr_test_q_hi_long
            # Calculate metric result
            [c_gu_long[j], s_gu_long[j], w_gu_long[j]] = metric_calculation(curr_test_target_long, intervals_gu_long)

        if "CQR_max" in algorithms:
            # time seq Union Bound
            alpha_err = np.max(err , axis=1)
            cal_scores = {0: np.sort(alpha_err, 0)[::-1]}
            nc = np.sort(cal_scores[0], 0)
            test_err = np.vstack([nc[index], nc[index]])
            intervals_ts = np.zeros((num_of_tests, T, 2))
            intervals_ts[:, :, 0] = curr_test_q_low - np.tile(test_err[0, :], (num_of_tests, T))
            intervals_ts[:, :, 1] = curr_test_q_hi + np.tile(test_err[1, :], (num_of_tests, T))

            # Calculate metric result
            [c_ts[j], s_ts[j], w_ts[j]] = metric_calculation(curr_test_target, intervals_ts)

        # time seq non-Union alpha

        if "CQR_alpha" in algorithms:
            alpha_err = np.max(err * alphas, axis=1)
            cal_scores = {0: np.sort(alpha_err, 0)[::-1]}
            nc = np.sort(cal_scores[0], 0)
            D_cp = [nc[index] / a for a in alphas]
            test_err = np.vstack([D_cp, D_cp])
            intervals_ts_alpha = np.zeros((num_of_tests, T, 2))
            intervals_ts_alpha[:, :, 0] = curr_test_q_low - np.tile(test_err[0, :], (num_of_tests, 1))
            intervals_ts_alpha[:, :, 1] = curr_test_q_hi + np.tile(test_err[1, :], (num_of_tests, 1))
            # Calculate metric result
            [c_ts_alpha[j], s_ts_alpha[j], w_ts_alpha[j]] = metric_calculation(curr_test_target, intervals_ts_alpha)

        if "time-series non Union" in algorithms:
            shuffled_idx_all = torch.randperm(curr_train_target.size(0))
            # shuffled_idx_alpha = shuffled_idx_all[:100]
            # curr_target_alpha = train_target.index_select(0, shuffled_idx_alpha)
            # curr_output_alpha = train_KF_output.index_select(0, shuffled_idx_alpha)
            # # alphas calibration
            # test_target_list = curr_target_alpha.squeeze().tolist()
            # x_array_test_list = curr_output_alpha.tolist()
            # R_vals = [[math.sqrt((test_target_list[i][j] - x_array_test_list[i][j]) ** 2) for j in
            #            range(len(test_target_list[i]))] for i in range(len(curr_target_alpha))]
            #
            # m = optimzeTimeAlphasKKTNoMaxLowerBound(R_vals, alpha, 100000)
            # alphas_v1 = []
            # for v in m.getVars():
            #     if "alphas" in v.varName:
            #         alphas_v1.append(v.x)
            #     if "q" in v.varName:
            #         # print(v.x)
            #         print("obj: " + str(v.x))
            ## Naive cp
            # shuffled_idx_training = shuffled_idx_all[100:]
            curr_train_target_training = train_target[:100, :]#.index_select(0, shuffled_idx_training)
            curr_train_output_training = curr_train_output[:100, :]#.index_select(0, shuffled_idx_training)
            test_target_list = curr_train_target_training.squeeze().tolist()
            x_array_test_list = curr_train_output_training.tolist()
            # calculate region
            R_vals = [
                max([alphas_v1[j] * math.sqrt((test_target_list[i][j] - x_array_test_list[i][j]) ** 2) for j in
                     range(len(test_target_list[i]))]) for i in range(len(test_target_list))]

            R_vals.sort()
            R_vals.append(max(R_vals))
            cal_scores_TS = {0: np.sort(R_vals, 0)[::-1]}
            nc = np.sort(cal_scores_TS[0], 0)

            index_TS = int(np.ceil((1 - alpha) * (nc.shape[0] + 1))) - 1
            index_TS = min(max(index_TS, 0), nc.shape[0] - 1)

            D_cp = R_vals[index_TS]
            D_cp_all = [D_cp / a for a in alphas_v1]
            intervals_ts_only = np.zeros((num_of_tests, T, 2))

            D_cp_all = np.ones(T) * D_cp_all
            intervals_ts_only[:, :, 0] = curr_test_output - np.tile(D_cp_all, (num_of_tests, 1))
            intervals_ts_only[:, :, 1] = curr_test_output + np.tile(D_cp_all, (num_of_tests, 1))

            # Calculate metric result
            [c_ts_only[j], s_ts_only[j], w_ts_only[j]] = metric_calculation(curr_test_target, intervals_ts_only)
        if "Union-TS" in algorithms:
            ## Naive cp
            test_target_list = curr_train_target.squeeze().tolist()
            x_array_test_list = curr_train_output.tolist()
            alphas_v2 = 1#;/T
            # calculate region
            R_vals = [
                max([alphas_v2 * math.sqrt((test_target_list[i][j] - x_array_test_list[i][j]) ** 2) for j in
                     range(len(test_target_list[i]))]) for i in range(len(test_target_list))]

            R_vals.sort()
            R_vals.append(max(R_vals))
            cal_scores_TS = {0: np.sort(R_vals, 0)[::-1]}
            nc_TS = np.sort(cal_scores_TS[0], 0)
            index_TS = int(np.ceil((1 - alpha) * (nc_TS.shape[0] + 1))) - 1
            index_TS = min(max(index_TS, 0), nc_TS.shape[0] - 1)
            D_cp = R_vals[index_TS]

            D_cp_all = D_cp / alphas_v2
            intervals_ts_only_union = np.zeros((num_of_tests, T, 2))

            D_cp_all = np.ones(T) * D_cp_all
            intervals_ts_only_union[:, :, 0] = curr_test_output - np.tile(D_cp_all, (num_of_tests, 1))
            intervals_ts_only_union[:, :, 1] = curr_test_output + np.tile(D_cp_all, (num_of_tests, 1))

            # Calculate metric result
            [c_ts_only_union[j], s_ts_only_union[j], w_ts_only_union[j]] = metric_calculation(curr_test_target, intervals_ts_only_union)


        # if plot_flag:
        #     if j == 0:
        #
        #         plt.plot(curr_test_target[0, 0, :], "bo")
        #         # plt.fill_between(
        #         #     np.arange(T), intervals[0,:, 0], intervals[0, :, 1], alpha=0.2, color="#0072B2",
        #         #     label="Interval CQKF-sw")
        #
        #         plt.fill_between(
        #             np.arange(T), intervals_ts[0,:, 0], intervals_ts[0, :, 1], alpha=0.2, color="#0072B2",
        #             label="Interval CQKF-tw")
        #         # plt.fill_between(
        #         #     np.arange(T), intervals_long[0,:, 0], intervals_long[0, :, 1], alpha=0.2, color="#009E73",
        #         #     label="Interval CQKF-Bonf-sw")
        #         # plt.fill_between(
        #         #     np.arange(T), intervals_ts_only[0,:, 0], intervals_ts_only[0, :, 1], alpha=0.2, color="#CC79A7",
        #         #     label="Interval Residuals-LCP")
        #
        #         plt.fill_between(
        #             np.arange(T), intervals_ts_only_union[0,:, 0], intervals_ts_only_union[0, :, 1], alpha=0.2, color="#E69F00",
        #             label="Interval Residuals-tw")
        #         # plt.fill_between(
        #         #     np.arange(T), intervals_gu_long[0,:, 0], intervals_ts_only[0, :, 1], alpha=0.2, color="#56B4E9",
        #         #     label="Pred. interval Gauss-Bonf-sw")
        #         # plt.fill_between(
        #         #     np.arange(T), intervals_gu[0, :, 0], intervals_gu[0, :, 1], alpha=0.2, color="#F0E442",
        #         #     label="Interval Gauss-sw")
        #         plt.xlabel("Sequence")
        #         plt.ylabel("Values and prediction intervals")
        #         plt.legend(loc="best")
        #         plt.grid()
        #         # plt.title("CQR")
        #         plt.show()

    c_pav_mean = np.mean(np.mean(c_pav, 1), 0) * 100
    s_pav_mean = np.mean(s_pav) * 100
    w_pav_mean = np.mean(np.mean(np.diff(w_pav), 1), 0)

    c_ts_alpha_mean = np.mean(np.mean(c_ts_alpha, 1), 0)*100
    s_ts_alpha_mean = np.mean(s_ts_alpha)*100
    w_ts_alpha_mean = np.mean(np.mean(np.diff(w_ts_alpha), 1), 0)


    c_gu_mean = np.mean(np.mean(c_gu, 1), 0)*100
    s_gu_mean = np.mean(s_gu)*100
    w_gu_mean = np.mean(np.mean(np.diff(w_gu), 1), 0)

    c_mean = np.mean(np.mean(c, 1), 0)*100
    s_mean = np.mean(s)*100
    w_mean = np.mean(np.mean(np.diff(w), 1), 0)

    c_mean_trained = np.mean(np.mean(c_trained, 1), 0)*100
    s_mean_trained = np.mean(s_trained)*100
    w_mean_trained = np.mean(np.mean(np.diff(w_trained), 1), 0)

    c_mean_cqkf_trained = np.mean(np.mean(c_cqkf_trained, 1), 0)*100
    s_mean_cqkf_trained = np.mean(s_cqkf_trained)*100
    w_mean_cqkf_trained = np.mean(np.mean(np.diff(w_cqkf_trained), 1), 0)

    c_mean_cqr_tw = np.mean(np.mean(c_cqr_tw, 1), 0)*100
    s_mean_cqr_tw  = np.mean(s_cqr_tw)*100
    w_mean_cqr_tw  = np.mean(np.mean(np.diff(w_cqr_tw), 1), 0)

    c_mean_cqkf_tw = np.mean(np.mean(c_cqkf_tw, 1), 0)*100
    s_mean_cqkf_tw = np.mean(s_cqkf_tw)*100
    w_mean_cqkf_tw = np.mean(np.mean(np.diff(w_cqkf_tw), 1), 0)

    c_mean_long = np.mean(np.mean(c_long, 1), 0)*100
    s_mean_long = np.mean(s_long)*100
    w_mean_long = np.mean(np.mean(np.diff(w_long), 1), 0)

    c_gu_mean_long = np.mean(np.mean(c_gu_long, 1), 0)*100
    s_gu_mean_long = np.mean(s_gu_long)*100
    w_gu_mean_long = np.mean(np.mean(np.diff(w_gu_long), 1), 0)

    c_ts_mean = np.mean(np.mean(c_ts,1), 0)*100
    s_ts_mean = np.mean(s_ts)*100
    w_ts_mean = np.mean(np.mean(np.diff(w_ts),1), 0)
    #
    c_ts_only_mean = np.mean(np.mean(c_ts_only,1), 0)*100
    s_ts_only_mean = np.mean(s_ts_only)*100
    w_ts_only_mean = np.mean(np.mean(np.diff(w_ts_only),1), 0)

    c_ts_only_union_mean = np.mean(np.mean(c_ts_only_union,1), 0)*100
    s_ts_only_union_mean = np.mean(s_ts_only_union)*100
    w_ts_only_union_mean = np.mean(np.mean(np.diff(w_ts_only_union),1), 0)

    c_dist_split_mean = np.mean(np.mean(c_dist_split,1), 0)*100
    s_dist_split_mean = np.mean(s_dist_split)*100
    w_dist_split_mean = np.mean(np.mean(np.diff(w_dist_split),1), 0)
    ## Insert to file
    writer = ScenarioMetricsWriter(scenario=scenario, outdir="outputs")
    writer.append(algo="CGKF-TjW", snr_db=snr,
                  C_mean=c_ts_mean.mean(), C_svd=c_ts_mean.std(), S=s_ts_mean, WI_mean=w_ts_mean.mean(), WI_svd=w_ts_mean.std())
    writer.append(algo="CGKF-sw", snr_db=snr,
                  C_mean=c_mean.mean(), C_svd=c_mean.std(), S=s_mean, WI_mean=w_mean.mean(), WI_svd=w_mean.std())
    writer.append(algo="LCP-TS", snr_db=snr,
                  C_mean=c_ts_only_mean.mean(), C_svd=c_ts_only_mean.std(), S=s_ts_only_mean, WI_mean=w_ts_only_mean.mean(), WI_svd=w_ts_only_mean.std())
    writer.append(algo="CGKF-Bonf-sw", snr_db=snr,
                  C_mean=c_mean_long.mean(), C_svd=c_mean_long.std(), S=s_mean_long, WI_mean=w_mean_long.mean(), WI_svd=w_mean_long.std())
    writer.append(algo="KF-Gauss", snr_db=snr,
                  C_mean=c_gu_mean.mean(), C_svd=c_gu_mean.std(), S=s_gu_mean, WI_mean=w_gu_mean.mean(), WI_svd=w_gu_mean.std())
    writer.append(algo="KF-Gauss-Bonf", snr_db=snr,
                  C_mean=c_gu_mean_long.mean(), C_svd=c_gu_mean_long.std(), S=s_gu_mean_long, WI_mean=w_gu_mean_long.mean(), WI_svd=w_gu_mean_long.std())
    writer.append(algo="Union-TS", snr_db=snr,
                  C_mean=c_ts_only_union_mean.mean(), C_svd=c_ts_only_union_mean.std(), S=s_ts_only_union_mean, WI_mean=w_ts_only_union_mean.mean(), WI_svd=w_ts_only_union_mean.std())
    writer.append(algo="CQKF-sw", snr_db=snr,
                  C_mean=c_mean_cqkf_trained.mean(), C_svd=c_mean_cqkf_trained.std(), S=s_mean_cqkf_trained, WI_mean=w_mean_cqkf_trained.mean(), WI_svd=w_mean_cqkf_trained.std())
    writer.append(algo="CQR-sw", snr_db=snr,
                  C_mean=c_mean_trained.mean(), C_svd=c_mean_trained.std(), S=s_mean_trained, WI_mean=w_mean_trained.mean(), WI_svd=w_mean_trained.std())
    writer.append(algo="CQKF-Tjw", snr_db=snr,
                  C_mean=c_mean_cqkf_tw.mean(), C_svd=c_mean_cqkf_tw.std(), S=s_mean_cqkf_tw, WI_mean=w_mean_cqkf_tw.mean(), WI_svd=w_mean_cqkf_tw.std())
    writer.append(algo="CQR-Tjw", snr_db=snr,
                  C_mean=c_mean_cqr_tw.mean(), C_svd=c_mean_cqr_tw.std(), S=s_mean_cqr_tw, WI_mean=w_mean_cqr_tw.mean(), WI_svd=w_mean_cqr_tw.std())

    # if "CQR_max" in algorithms:
    #     mean_pr_cp_ts = s_ts_mean
    #     plt.plot(c_ts_mean,marker='o', label=f'CQKF-tw (TrjFail={mean_pr_cp_ts:.3f} %)')
    # if 0:#"CQR_alpha" in algorithms:
    #     mean_pr_cp_ts = s_ts_alpha_mean
    #     plt.plot(c_ts_alpha_mean, label=f'CQKF-TjW-alpha (TrjFail={mean_pr_cp_ts:.3f} %)')
    #
    # if "time-series non Union" in algorithms:
    #     mean_pr_cp_ts = s_ts_only_mean
    #     plt.plot(c_ts_only_mean, marker='s', label=f'Residuals-LCP (TrjFail={mean_pr_cp_ts:.3f} %)')
    # if "Union-TS" in algorithms:
    #     mean_pr_cp_ts = s_ts_only_union_mean
    #     plt.plot(c_ts_only_union_mean, label=f'Residuals-tw (TrjFail={mean_pr_cp_ts:.3f} %)')
    #
    # if "CQR" in algorithms:
    #     mean_pr_cqr = s_mean
    #     plt.plot(c_mean,marker='o', label=f'CQKF-sw (TrjFail={mean_pr_cqr:.3f} %)')
    # if "CQR_trained" in algorithms:
    #     mean_pr_cqr_trained = s_mean_trained
    #     plt.plot(c_mean_trained,marker='o', label=f'CQKF-trained-sw (TrjFail={mean_pr_cqr_trained:.3f} %)')
    # if "CQKF-Bonf-sw" in algorithms:
    #     mean_pr_cqr = s_mean_long
    #     plt.plot(c_mean_long, label=f'CQKF-Bonf-sw (TrjFail={mean_pr_cqr:.3f} %)')
    # if "Gaussian only" in algorithms:
    #     mean_pr_cp_gu = s_gu_mean
    #     plt.plot(c_gu_mean,marker='^', label=f'Gauss-sw (TrjFail={mean_pr_cp_gu:.3f} %)')
    # if "Gaussian only long" in algorithms:
    #         mean_pr_cp_gu = s_gu_mean_long
    #         plt.plot(c_gu_mean_long,marker='^', label=f'Gauss-Bonf-sw (TrjFail={mean_pr_cp_gu:.3f} %)')
    # if 0:#"CQR_r" in algorithms:
    #     plt.plot(c_pav_mean, label=f'CQKF-sw-r (mean={s_pav_mean:.3f} %)')
    # if "dist-split" in algorithms:
    #     plt.plot(c_dist_split_mean, label=f'dist-split (mean={s_dist_split_mean:.3f} %)')
    # plt.xlabel("Sequence")
    # plt.ylabel("Error %")
    # plt.title(f"1/r2 [dB]: {title_value:.2f}")
    # plt.legend()
    # plt.grid()
    # plt.show()
    #
    # if "CQR_max" in algorithms:
    #     mean_pr_cp_ts = w_ts_mean.mean()
    #     plt.plot(w_ts_mean, marker='o', label=f'CQKF-tw (mean width={mean_pr_cp_ts:.3f} )')
    # if "CQR_alpha" in algorithms:
    #     mean_pr_cp_ts = w_ts_alpha_mean.mean()
    #     plt.plot(w_ts_alpha_mean, label=f'CQKF-TjW-alpha (mean width={mean_pr_cp_ts:.3f} )')
    # if "time-series non Union" in algorithms:
    #     mean_pr_cp_ts = w_ts_only_mean.mean()
    #     plt.plot(w_ts_only_mean, marker='s' , label=f'Residuals-LCP (mean width={mean_pr_cp_ts:.3f})')
    # if "Union-TS" in algorithms:
    #     mean_pr_cp_ts = w_ts_only_union_mean.mean()
    #     plt.plot(w_ts_only_union_mean, label=f'Residuals-tw (mean width={mean_pr_cp_ts:.3f})')
    # if "CQR" in algorithms:
    #     mean_pr_cqr = w_mean.mean()
    #     plt.plot(w_mean,marker='o', label=f'CQKF-sw (mean width={mean_pr_cqr:.3f})')
    # if "CQR_trained" in algorithms:
    #     mean_pr_cqr_trained = w_mean_trained.mean()
    #     plt.plot(w_mean_trained,marker='o', label=f'CQKF-trained-w (mean width={mean_pr_cqr_trained:.3f})')
    # if "CQKF-Bonf-sw" in algorithms:
    #     mean_pr_cqr = w_mean_long.mean()
    #     plt.plot(w_mean_long, label=f'CQKF-Bonf-sw (mean width={mean_pr_cqr:.3f})')
    # if "Gaussian only" in algorithms:
    #     mean_pr_cp_gu = w_gu_mean.mean()
    #     plt.plot(w_gu_mean, marker='^',label=f'Gauss-sw  (mean width={mean_pr_cp_gu:.3f})')
    # if "Gaussian only long" in algorithms:
    #     mean_pr_cp_gu = w_gu_mean_long.mean()
    #     plt.plot(w_gu_mean_long, marker='^', label=f'Gauss-Bonf-sw  (mean width={mean_pr_cp_gu:.3f})')
    # if "CQR_r" in algorithms:
    #     mean_pr_cp_gu = w_pav_mean.mean()
    #     plt.plot(w_pav_mean, label=f'CQKF-sw-r (mean width={mean_pr_cp_gu:.3f})')
    # if "dist-split" in algorithms:
    #     mean_pr_cp_gu = w_dist_split_mean.mean()
    #     plt.plot(w_dist_split_mean, label=f'dist-split (mean width={mean_pr_cp_gu:.3f})')
    # plt.xlabel("Sequence")
    # plt.ylabel("Interval Width")
    # plt.title(f"1/r2 [dB]: {title_value:.2f}")
    # plt.legend()
    # plt.grid()
    # plt.show()
    return c_mean, s_mean, w_mean

class ScenarioMetricsWriter:
    """
    Per-scenario CSV with multiple algorithms.
    Columns: [scenario, algo, snr_db, C_mean, C_svd, S, WI_mean, WI_svd]

    Usage:
        writer = ScenarioMetricsWriter(scenario="LC-Gauss", outdir="outputs")
        writer.append(algo="CQKF-sw", snr_db=10,
                      C_mean=..., C_svd=..., S=..., WI_mean=..., WI_svd=...)
    """
    def __init__(self, scenario: str, outdir: str = "outputs", prefix: str = "metrics"):
        self.scenario = scenario
        self.outdir = outdir
        self.prefix = prefix
        self.fieldnames = ["scenario", "algo", "snr_db", "C_mean", "C_svd", "S", "WI_mean", "WI_svd"]

        os.makedirs(self.outdir, exist_ok=True)
        safe_scenario = re.sub(r"[^A-Za-z0-9._-]+", "_", scenario)
        self.path = os.path.join(self.outdir, f"{self.prefix}_{safe_scenario}.csv")

        # Create the file with header if new/empty
        needs_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        if needs_header:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def append(self, *, algo: str, snr_db: float,
               C_mean: float, C_svd: float, S: float, WI_mean: float, WI_svd: float) -> None:
        row = {
            "scenario": self.scenario,
            "algo": algo,
            "snr_db": snr_db,
            "C_mean": C_mean,
            "C_svd": C_svd,
            "S": S,
            "WI_mean": WI_mean,
            "WI_svd": WI_svd,
        }
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)

    def filepath(self) -> str:
        return self.path