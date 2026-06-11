import torch
import torch.nn as nn
import random
import time
from Plot import Plot
import time
import numpy as np
import matplotlib.pyplot as plt
from KalmanFilter_EstOnly_test import KFTest
from CP_test import compute_gaussian_quantiles, clalibration_residual_quantiles

def gaussian_nll(target, predicted_mean, predicted_var):
    """Gaussian Negative Log Likelihood (assuming diagonal covariance)"""
    predicted_var += 1e-12
    N_T = predicted_mean.size(1)
    mahal = torch.square(target[:, 0:N_T] - predicted_mean) / torch.abs(predicted_var)
    element_wise_nll = 0.5 * (torch.log(torch.abs(predicted_var)) + torch.log(torch.tensor(2 * torch.pi)) + mahal)
    sample_wise_error = torch.sum(element_wise_nll, dim=-1)
    return torch.mean(sample_wise_error)


def pinball_loss(target, preds, quantiles):
    """ Compute the pinball loss

    Parameters
    ----------
    preds : pytorch tensor of estimated labels (n)
    target : pytorch tensor of true labels (n)

    Returns
    -------
    loss : cost function value

    """
    assert not target.requires_grad
    # assert preds.size(0) == target.size(0)
    losses = []

    for i, q in enumerate(quantiles):
        errors = target - preds[i, :]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))

    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss


class Pipeline_KF:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName
        self.PipelineName = self.folderName + "pipeline_" + self.modelName

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):
        self.model = model

    def setTrainingParams(self, n_Epochs, n_Batch, learningRate, weightDecay):
        self.N_Epochs = n_Epochs  # Number of Training Epochs
        self.N_B = n_Batch  # Number of Samples in Batch
        self.learningRate = learningRate  # Learning Rate
        self.weightDecay = weightDecay  # L2 Weight Regularization - Weight Decay

        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

    def NNTrain(self, n_Examples, train_input, train_target, n_CV, cv_input, cv_target):
        self.Beta = 0.1
        self.N_E = n_Examples
        self.N_CV = n_CV

        MSE_cv_linear_batch = torch.empty([self.N_CV])
        self.MSE_cv_linear_epoch = torch.empty([self.N_Epochs])
        self.MSE_cv_dB_epoch = torch.empty([self.N_Epochs])

        # Loss for MSE part
        MSE_mse_cv_linear_batch = torch.empty([self.N_CV])
        self.MSE_mse_cv_linear_epoch = torch.empty([self.N_Epochs])
        self.MSE_mse_cv_dB_epoch = torch.empty([self.N_Epochs])
        # Loss for PD part
        MSE_pd_cv_linear_batch = torch.empty([self.N_CV])
        self.MSE_pd_cv_linear_epoch = torch.empty([self.N_Epochs])
        self.MSE_pd_cv_dB_epoch = torch.empty([self.N_Epochs])

        MSE_train_linear_batch = torch.empty([self.N_B])
        self.MSE_train_linear_epoch = torch.empty([self.N_Epochs])
        self.MSE_train_dB_epoch = torch.empty([self.N_Epochs])

        # Loss for MSE part
        MSE_mse_train_linear_batch = torch.empty([self.N_B])
        self.MSE_mse_train_linear_epoch = torch.empty([self.N_Epochs])
        self.MSE_mse_train_dB_epoch = torch.empty([self.N_Epochs])
        # Loss for PD part
        MSE_pd_train_linear_batch = torch.empty([self.N_B])
        self.MSE_pd_train_linear_epoch = torch.empty([self.N_Epochs])
        self.MSE_pd_train_dB_epoch = torch.empty([self.N_Epochs])


        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for ti in range(0, self.N_Epochs):

            #################################F
            ### Validation Sequence Batch ###
            #################################
            # BETA = self.Beta# * (ti / self.N_Epochs)

            # Cross Validation Mode
            self.model.eval()

            for j in range(0, self.N_CV):
                y_cv = cv_input[j, :, :]
                self.model.InitSequence(self.ssModel.m1x_0)

                x_out_cv = torch.empty(self.ssModel.m, self.ssModel.T)
                q_cv = torch.empty(2, self.ssModel.T)
                per_sample_miscovarage = torch.empty(self.ssModel.T)
                for t in range(0, self.ssModel.T):
                    x_out_cv[:, t], q_cv[:, t] = self.model(y_cv[:, t])
                LOSS_MSE = self.loss_fn(x_out_cv, cv_target[j, :, :self.ssModel.T])
                LOSS_PD = pinball_loss(cv_target[j, :, :self.ssModel.T], q_cv, [0.05, 0.95])

                beta_eff = self.Beta#*(LOSS_PD.item()/(LOSS_MSE.item() + self.Beta*LOSS_PD.item()))
                LOSS = (1 - beta_eff) *LOSS_MSE + beta_eff * LOSS_PD

                # Compute Training Loss
                # Olga
                MSE_cv_linear_batch[j] = LOSS.item()
                MSE_mse_cv_linear_batch[j] = LOSS_MSE.item()
                MSE_pd_cv_linear_batch[j] = LOSS_PD.item()

            # Average
            self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
            self.MSE_mse_cv_linear_epoch[ti] = torch.mean(MSE_mse_cv_linear_batch)
            self.MSE_pd_cv_linear_epoch[ti] = torch.mean(MSE_pd_cv_linear_batch)
            self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])
            self.MSE_mse_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_mse_cv_linear_epoch[ti])
            self.MSE_pd_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_pd_cv_linear_epoch[ti])

            if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                self.MSE_cv_idx_opt = ti
                torch.save(self.model, self.modelFileName)

            ###############################
            ### Training Sequence Batch ###
            ###############################

            # Training Mode
            self.model.train()

            # Init Hidden State
            self.model.init_hidden()

            Batch_Optimizing_LOSS_sum = 0

            for j in range(0, self.N_B):
                n_e = random.randint(0, self.N_E - 1)

                y_training = train_input[n_e, :, :]
                self.model.InitSequence(self.ssModel.m1x_0)

                x_out_training = torch.empty(self.ssModel.m, self.ssModel.T)
                q_train = torch.empty(2, self.ssModel.T) #TODO
                # cov_out_training = torch.empty(self.ssModel.m, self.ssModel.T) #Olga
                for t in range(0, self.ssModel.T):
                    x_out_training[:, t], q_train[:, t] = self.model(y_training[:, t])
                    # covariance
                    # cov_out_training[:, t] = torch.diagonal(self.model.cov_post_byK_optA )#Olga
                # Compute Training Loss
                LOSS_MSE = self.loss_fn(x_out_training, train_target[n_e, :, :self.ssModel.T])
                LOSS_PD = pinball_loss(train_target[n_e, :, :self.ssModel.T], q_train, [0.05, 0.95])
                beta_eff = self.Beta #* (LOSS_PD.item() / (LOSS_MSE.item() + self.Beta * LOSS_PD.item()))
                LOSS = (1 - beta_eff) * LOSS_MSE + beta_eff * LOSS_PD

                MSE_train_linear_batch[j] = LOSS.item()
                MSE_mse_train_linear_batch[j] = LOSS_MSE.item()
                MSE_pd_train_linear_batch[j] = LOSS_PD.item()

                Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + LOSS

            # Average
            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
            self.MSE_mse_train_linear_epoch[ti] = torch.mean(MSE_mse_train_linear_batch)
            self.MSE_pd_train_linear_epoch[ti] = torch.mean(MSE_pd_train_linear_batch)

            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])
            self.MSE_mse_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_mse_train_linear_epoch[ti])
            self.MSE_pd_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_pd_train_linear_epoch[ti])

            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
            Batch_Optimizing_LOSS_mean.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :",
                  self.MSE_cv_dB_epoch[ti],
                  "[dB]")

            print(ti, "MSE mse Training :", self.MSE_mse_train_dB_epoch[ti], "[dB]", "MSE Validation :",
                  self.MSE_mse_cv_dB_epoch[ti],
                  "[dB]")
            print(ti, "MSE pd Training :", self.MSE_pd_train_dB_epoch[ti], "[dB]", "MSE Validation :",
                  self.MSE_pd_cv_dB_epoch[ti],
                  "[dB]")

            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

    def NNCalibrate_Predict(self,n_Calib, n_Trails, calib_input, calib_target, alpha):

        self.model = torch.load(self.modelFileName)

        self.model.eval()

        torch.no_grad()

        num_of_calib = int(round(n_Calib * 0.8)) # 80% of the data

        self.N_T = n_Calib - num_of_calib

        c = torch.empty(n_Trails, self.N_T, self.ssModel.T)
        s = torch.empty(n_Trails, self.N_T)
        w = torch.empty(n_Trails, self.N_T, self.ssModel.T, 2)

        # CQKF
        c_cqkf = torch.empty(n_Trails, self.N_T, self.ssModel.T)
        s_cqkf = torch.empty(n_Trails, self.N_T)
        w_cqkf = torch.empty(n_Trails, self.N_T, self.ssModel.T, 2)

        # for CQKF
        [x_array_train, sigma_array_train] = KFTest(self.ssModel, calib_input)
        [q_low, q_hi, _] = compute_gaussian_quantiles(x_array_train, sigma_array_train, alpha)
        for ii in range(0, n_Trails):
            err = torch.empty([num_of_calib, self.ssModel.T])
            err_cqkf = torch.empty([num_of_calib, self.ssModel.T])

            shuffled_idx = torch.randperm(calib_input.size(0))

            curr_calib_input = calib_input.index_select(0, shuffled_idx)
            curr_calib_target = calib_target.index_select(0, shuffled_idx)

            calibration_input = curr_calib_input[:num_of_calib]
            calibration_target = curr_calib_target[:num_of_calib]

            test_input = curr_calib_input[num_of_calib:]
            test_target = curr_calib_target[num_of_calib:]

            curr_q_low = torch.as_tensor(q_low).index_select(0, shuffled_idx)
            curr_q_hi = torch.as_tensor(q_hi).index_select(0, shuffled_idx)

            calibration_q_low = curr_q_low[:num_of_calib]
            calibration_q_hi = curr_q_hi[:num_of_calib]

            test_q_low = curr_q_low[num_of_calib:]
            test_q_hi = curr_q_hi[num_of_calib:]


            for j in range(0, num_of_calib):

                y_mdl_tst = calibration_input[j, :, :]

                self.model.InitSequence(self.ssModel.m1x_0)

                x_out_test = torch.empty(self.ssModel.m, self.ssModel.T)
                q_test = torch.empty(2, self.ssModel.T)

                for t in range(0, self.ssModel.T):
                    x_out_test[:, t], q_test[:, t] = self.model(y_mdl_tst[:, t])
                error_low = q_test[0, :] - calibration_target[j, :, :self.ssModel.T].squeeze()
                error_high = calibration_target[j, :, :self.ssModel.T].squeeze() - q_test[1, :]
                err[j, :] = torch.maximum(error_high, error_low)
                # CQKF error
                error_low_CQKF = calibration_q_low[j, :] - calibration_target[j, :, :self.ssModel.T].squeeze()
                error_high_CQKF = calibration_target[j, :, :self.ssModel.T].squeeze() - calibration_q_hi[j, :]
                err_cqkf[j, :] = torch.maximum(error_low_CQKF, error_high_CQKF)
            # Sort descending
            cal_scores = {0: torch.sort(err, dim=0, descending=True).values}
            cal_scores_cqkf = {0: torch.sort(err_cqkf, dim=0, descending=True).values}
            # Sort ascending again (like your second np.sort)
            self.nc = torch.sort(cal_scores[0], dim=0).values
            self.nc_cqkf = torch.sort(cal_scores_cqkf[0], dim=0).values

            index = int(np.ceil((1 - 0.05) * (self.nc.shape[0] + 1))) - 1 #TODO
            self.index = min(max(index, 0), self.nc.shape[0] - 1)

            ##########################################
            ###### PREDICT ###########################
            ##########################################

            test_err = torch.stack([self.nc[self.index, :], self.nc[self.index, :]], dim=0)
            test_err_cqkf = torch.stack([self.nc_cqkf[self.index, :], self.nc_cqkf[self.index, :]], dim=0)

            for j in range(0, self.N_T):

                y_mdl_tst = test_input[j, :, :]

                self.model.InitSequence(self.ssModel.m1x_0)

                x_out_test = torch.empty(self.ssModel.m, self.ssModel.T)
                q_test = torch.empty(2, self.ssModel.T)
                intervals = torch.zeros((self.ssModel.T, 2))
                intervals_cqkf = torch.zeros((self.ssModel.T, 2))
                for t in range(0, self.ssModel.T):
                    x_out_test[:, t], q_test[:, t] = self.model(y_mdl_tst[:, t])

                intervals[:, 0] = q_test[0, :] - test_err[0, :]
                intervals[:, 1] = q_test[1, :] + test_err[1, :]
                intervals_cqkf[:, 0] = test_q_low[j, :] - test_err_cqkf[0, :]
                intervals_cqkf[:, 1] = test_q_hi[j, :] + test_err_cqkf[1, :]
                c[ii, j, :] = torch.logical_or(test_target[j, :, :self.ssModel.T] < intervals[:, 0],
                                           test_target[j, :, :self.ssModel.T] > intervals[:, 1]).float()
                s[ii, j] = (torch.sum(c[ii, j, :], dim=-1) > 0).item()
                w[ii, j, :, :] = intervals

                c_cqkf[ii, j, :] = torch.logical_or(test_target[j, :, :self.ssModel.T] < intervals_cqkf[:, 0],
                                                test_target[j, :, :self.ssModel.T] > intervals_cqkf[:, 1]).float()
                s_cqkf[ii, j] = (torch.sum(c_cqkf[ii, j, :], dim=-1) > 0).item()
                w_cqkf[ii, j, :, :] = intervals_cqkf
                #### DEBUG ######
                if j == 0 and ii == 0:
                    plt.plot(test_target[j, :, :self.ssModel.T].detach().cpu().squeeze(), "bo")


                    # plt.fill_between(
                    #     np.arange(self.ssModel.T), intervals[:, 0].detach().cpu(), intervals[:, 1].detach().cpu(),
                    #     alpha=0.2, color="#0072B2",
                    #     label="Interval CQR")

                    plt.fill_between(
                        np.arange(self.ssModel.T), q_test[0, :].detach().cpu(), q_test[1, :].detach().cpu(), alpha=0.2,
                        color="#E69F00",
                        label="Quantile Regression")

                    # plt.fill_between(
                    #     np.arange(self.ssModel.T), intervals_cqkf[:, 0].detach().cpu(),
                    #     intervals_cqkf[:, 1].detach().cpu(), alpha=0.2, color="#009E73",
                    #     label="CQKF")

                    plt.fill_between(
                        np.arange(self.ssModel.T), test_q_low[j, :], test_q_hi[j, :], alpha=0.2, color="#D55E00",
                        label="KF-Gauss")

                    plt.xlabel("Sequence")
                    plt.ylabel("Values and prediction intervals")
                    plt.legend(loc="best")
                    plt.grid()
                    plt.show()

        per_sample_miscoverage = c.mean() * 100
        per_trajectory_miscoverage = torch.mean(s) * 100
        w_mean = torch.diff(w).mean()
        print(f"Per-sample miscoverage: {per_sample_miscoverage:.2f}%")
        print(f"Per-trajectory miscoverage: {per_trajectory_miscoverage:.2f}%")
        print(f"Mean width: {w_mean.item():.2f}")

        per_sample_miscoverage = c_cqkf.mean() * 100
        per_trajectory_miscoverage = torch.mean(s_cqkf) * 100
        w_mean = torch.diff(w_cqkf).mean()
        print(f"Per-sample miscoverage cqkf: {per_sample_miscoverage:.2f}%")
        print(f"Per-trajectory miscoverage cqkf: {per_trajectory_miscoverage:.2f}%")
        print(f"Mean width cqkf: {w_mean.item():.2f}")



    def NNTest(self, n_Test, test_input, test_target):

        self.N_T = n_Test

        self.MSE_test_linear_arr = torch.empty([self.N_T])

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        self.model = torch.load(self.modelFileName)

        self.model.eval()

        torch.no_grad()

        start = time.time()

        for j in range(0, self.N_T):

            y_mdl_tst = test_input[j, :, :]

            self.model.InitSequence(self.ssModel.m1x_0)

            x_out_test = torch.empty(self.ssModel.m, self.ssModel.T)
            q_test = torch.empty(2, self.ssModel.T)

            for t in range(0, self.ssModel.T):
                x_out_test[:, t], q_test[:, t] = self.model(y_mdl_tst[:, t])

            self.MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j, :, :]).item()

        end = time.time()
        t = end - start

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Standard deviation
        self.MSE_test_dB_std = torch.std(self.MSE_test_linear_arr, unbiased=True)
        self.MSE_test_dB_std = 10 * torch.log10(self.MSE_test_dB_std)

        # Print MSE Cross Validation
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        str = self.modelName + "-" + "STD Test:"
        print(str, self.MSE_test_dB_std, "[dB]")
        # Print Run Time
        print("Inference Time:", t)

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_test]

    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_Epochs, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)