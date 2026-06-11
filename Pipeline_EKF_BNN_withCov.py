import torch
import torch.nn as nn
import random
import time
from Plot import Plot
import time


def custom_bnn_loss(batch_x, x_hat, cov_hat_bnn, regularization):
    """
    Computes a custom loss composed of:
    - L1: MSE between true and predicted values (excluding first dim)
    - L2: Error between squared error and predicted covariance
    - L3: Regularization term (e.g., from ConcreteDropout)

    Args:
        batch_x (Tensor): Ground-truth data, shape [batch_size, seq_len, dim]
        x_hat (Tensor): Model predictions, shape [batch_size, seq_len, dim]
        cov_hat_bnn (Tensor): Predicted uncertainty (variance), same shape as x_hat
        regularization (Tensor): Regularization per sample, same shape as x_hat
    Returns:
        total_loss (Tensor): Scalar loss
        loss_L1 (Tensor)
        loss_L2 (Tensor)
        loss_L3 (Tensor)
    """
    # L1: MSE
    N_T = x_hat.size(1)
    loss_L1 = torch.mean((batch_x[:, 1:N_T] - x_hat[:, 1:N_T]).pow(2))

    # L2: Calibration loss between squared error and predicted covariance
    squared_error = (batch_x[:, 1:N_T] - x_hat[:, 1:N_T]).pow(2)
    loss_L2 = torch.sum(torch.abs(squared_error - cov_hat_bnn[:, 1:N_T]))

    # L3: Regularization term (e.g., from ConcreteDropout)
    loss_L3 = torch.sum(regularization[:, 1:])

    return loss_L1, loss_L2, loss_L3
class Pipeline_EKF:

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

    def setTrainingParams(self, n_Epochs, n_Batch, learningRate, weightDecay, num_of_mc_iterations):
        self.N_Epochs = n_Epochs  # Number of Training Epochs
        self.N_B = n_Batch  # Number of Samples in Batch
        self.learningRate = learningRate  # Learning Rate
        self.weightDecay = weightDecay  # L2 Weight Regularization - Weight Decay
        self.num_of_mc_iterations = num_of_mc_iterations #
        self.beta = 0.01 #TODO # for training loss BNN

        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')
        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

    def NNTrain(self, n_Examples, train_input, train_target, n_CV, cv_input, cv_target):

        self.N_E = n_Examples
        self.N_CV = n_CV

        MSE_cv_linear_batch = torch.empty([self.N_CV])
        self.MSE_cv_linear_epoch = torch.empty([self.N_Epochs])
        self.MSE_cv_dB_epoch = torch.empty([self.N_Epochs])


        MSE_train_linear_batch = torch.empty([self.N_B])
        self.MSE_train_linear_epoch = torch.empty([self.N_Epochs])
        self.MSE_train_dB_epoch = torch.empty([self.N_Epochs])

        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for ti in range(0, self.N_Epochs):


            ###############################
            ### Training Sequence Batch ###
            ###############################

            # Training Mode
            self.model.train()

            # Init Hidden State
            self.model.init_hidden()

            Batch_Optimizing_LOSS_sum_L1 = 0
            Batch_Optimizing_LOSS_sum_L2 = 0
            Batch_Optimizing_LOSS_sum_L3 = 0

            for j in range(0, self.N_B):
                torch.autograd.set_detect_anomaly(True)
                n_e = random.randint(0, self.N_E - 1)

                y_training = train_input[n_e, :, :]
                x_out_training = torch.empty(self.ssModel.m, self.ssModel.T, self.num_of_mc_iterations)
                regularization = torch.empty(2, self.ssModel.T, self.num_of_mc_iterations) #TODO
                cov_out_training = torch.empty(self.ssModel.m, self.ssModel.T)

                self.model.InitSequence(self.ssModel.m1x_0.reshape(-1, 1).repeat(1,self.num_of_mc_iterations),self.ssModel.T) # repmat for MC iterations
                cov_out_training_tmp = torch.zeros(self.ssModel.m)
                gt_cov_training_tmp = torch.zeros(self.ssModel.m)
                cov_out_training_list = []

                for t in range(0, self.ssModel.T):

                    self.model.InitMonteCarlo()

                    for mc_idx in range(0, self.num_of_mc_iterations):
                        x_out_training[:, t, mc_idx], regularization[:, t, mc_idx] = self.model(y_training[:, t],gt_cov_training_tmp) #Olga change
                        if torch.isnan(x_out_training[0, t, mc_idx]):
                            print("x_out_training[0, %d, %d] = nan", t, mc_idx)
                    cov_out_training_tmp = torch.var(x_out_training[:, t, :].detach(), dim=-1)  # or True if needed
                    gt_cov_training_tmp = (train_target[n_e,:, t] - torch.mean(x_out_training[:,t,:], dim = -1)).pow(2).detach().clone()
                    cov_out_training_list.append(cov_out_training_tmp)

                cov_out_training = torch.stack(cov_out_training_list, dim=1)
                # Compute Training Loss
                L1,L2,L3 = custom_bnn_loss(train_target[n_e, :, :], torch.mean(x_out_training, dim = -1), cov_out_training, torch.mean(regularization,dim = -1))

                # MSE_train_linear_batch[j] = L1.item()
                Batch_Optimizing_LOSS_sum_L1 = Batch_Optimizing_LOSS_sum_L1 + L1
                Batch_Optimizing_LOSS_sum_L2 = Batch_Optimizing_LOSS_sum_L2 + L2
                Batch_Optimizing_LOSS_sum_L3 = Batch_Optimizing_LOSS_sum_L3 + L3


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
            BETA = self.beta * (ti / self.N_Epochs)
            Batch_Optimizing_LOSS_mean = (1 - BETA) * (Batch_Optimizing_LOSS_sum_L1 / self.N_B) + BETA * Batch_Optimizing_LOSS_sum_L2 + Batch_Optimizing_LOSS_sum_L3

            # Average

            self.MSE_train_linear_epoch[ti] = Batch_Optimizing_LOSS_mean.item()
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            Batch_Optimizing_LOSS_mean.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()


            #################################
            ### Validation Sequence Batch ###
            #################################
            self.model.eval()

            for j in range(0, self.N_CV):
                y_cv = cv_input[j, :, :]

                x_out_cv = torch.empty(self.ssModel.m, self.ssModel.T,self.num_of_mc_iterations)

                self.model.InitSequence(self.ssModel.m1x_0.reshape(-1, 1).repeat(1, self.num_of_mc_iterations), self.ssModel.T) # repmat for MC iterations


                cov_out_cv_tmp = torch.empty(self.ssModel.m)
                gt_cov_cv_tmp = torch.empty(self.ssModel.m)
                for t in range(0, self.ssModel.T):

                    self.model.InitMonteCarlo()

                    for mc_idx in range(0, self.num_of_mc_iterations):
                        x_out_cv[:, t, mc_idx], _ = self.model(y_cv[:, t], gt_cov_cv_tmp)

                    cov_out_cv_tmp = torch.var(x_out_cv[:, t, :].clone(), dim=-1)  # or True if needed
                    gt_cov_cv_tmp = (cv_target[j, :, t] - torch.mean(x_out_cv[:, t, :], dim=-1)).pow(2).detach().clone()
                # Compute Training Loss
                MSE_cv_linear_batch[j] = self.loss_fn(torch.mean(x_out_cv, dim=-1), cv_target[j, :, 0:self.ssModel.T,]).item()


            # Average
            self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
            self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

            if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                self.MSE_cv_idx_opt = ti
                torch.save(self.model, self.modelFileName)

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :",
                  self.MSE_cv_dB_epoch[ti],
                  "[dB]")

            # if (ti > 1):
            #     d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
            #     d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
            #     print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

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

            x_out_test = torch.empty(self.ssModel.m, self.ssModel.T, self.num_of_mc_iterations)

            cov_out_test_tmp = torch.empty(self.ssModel.m)
            gt_cov_out_test_tmp = torch.empty(self.ssModel.m)

            self.model.InitSequence(self.ssModel.m1x_0.reshape(-1, 1).repeat(1, self.num_of_mc_iterations), self.ssModel.T) # repmat for MC iterations

            for t in range(0, self.ssModel.T):

                self.model.InitMonteCarlo()

                for mc_idx in range(0, self.num_of_mc_iterations):
                    x_out_test[:, t, mc_idx], _ = self.model(y_mdl_tst[:, t], gt_cov_out_test_tmp)

                cov_out_test_tmp = torch.var(x_out_test[:, t, :].clone(), dim=-1)
                gt_cov_out_test_tmp = (test_target[j, :, t] - torch.mean(x_out_test[:, t, :], dim=-1)).pow(2).clone().detach()

            self.MSE_test_linear_arr[j] = loss_fn(torch.mean(x_out_test, dim=-1), test_target[j, :, :self.ssModel.T]).item()

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