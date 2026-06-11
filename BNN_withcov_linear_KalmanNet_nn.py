"""# **Class: KalmanNet**"""

import torch
import torch.nn as nn
import torch.nn.functional as func
from ConcreteDropout import ConcreteDropout

class KalmanNetNN_E(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #############
    ### Build ###
    #############
    def Build(self, ssModel):

        self.InitSystemDynamics(ssModel.F, ssModel.H, ssModel.Q, ssModel.R)

        # Number of neurons in the 1st hidden layer
        H1_KNet = (ssModel.m + ssModel.n) * (10) * 8  #Olga

        # Number of neurons in the 2nd hidden layer
        H2_KNet = (ssModel.m * ssModel.n) * 1 * (4)  #Olga

        self.InitKGainNet(H1_KNet, H2_KNet)

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitKGainNet(self, H1, H2):
        # Input Dimensions
        D_in = self.m + self.n + self.m  # x(t-1), y(t), cov(t-1)

        # Output Dimensions
        D_out = self.m * self.n  # Kalman Gain

        ###################
        ### Input Layer ###
        ###################
        # Linear Layer
        self.KG_l1 = nn.Sequential(
            nn.Linear(D_in, H1, bias=True),
            nn.ReLU()
        )

        # ConcreteDropout
        self.KG_l1_con = ConcreteDropout(device=self.device)
        ###########
        ### GRU ###
        ###########
        # Input Dimension
        self.input_dim = H1
        # Hidden Dimension
        self.hidden_dim = (self.m * self.m + self.n * self.n) * 20
        # Number of Layers
        self.n_layers = 1
        # Batch Size
        self.batch_size = 1
        # Input Sequence Length
        self.seq_len_input = 1
        # Hidden Sequence Length
        self.seq_len_hidden = self.n_layers

        # batch_first = False
        # dropout = 0.1 ;

        # Initialize a Tensor for GRU Input
        # self.GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim)

        # Initialize a Tensor for Hidden State
        self.hn = torch.randn(self.seq_len_hidden, self.batch_size, self.hidden_dim).to(self.device, non_blocking = True)

        # Iniatialize GRU Layer
        self.rnn_GRU = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers)


        self.KG_l2 = nn.Sequential(
            torch.nn.Linear(self.hidden_dim, H2, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(H2, D_out, bias=True)
        )
        # ConcreteDropout
        self.KG_l2_con = ConcreteDropout(device=self.device)

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, F, H, Q, R):
        # Set State Evolution Matrix
        self.F = F.to(self.device,non_blocking = True)
        self.F_T = torch.transpose(F, 0, 1).to(self.device,non_blocking = True)
        self.m = self.F.size()[0]

        # Set Observation Matrix
        self.H = H.to(self.device,non_blocking = True)
        self.H_T = torch.transpose(H, 0, 1).to(self.device,non_blocking = True)
        self.n = self.H.size()[0]

        self.Q = Q
        self.R = R


    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0):

        self.m1x_prior = M1_0.to(self.device,non_blocking = True)

        self.m1x_posterior = M1_0.to(self.device,non_blocking = True)

        self.state_process_posterior_0 = M1_0.to(self.device,non_blocking = True)

        # New for Initialization
        self.state_process_prior_0 = self.state_process_posterior_0.detach()
        self.obs_process_0 =  self.state_process_posterior_0.detach()
        self.m1x_prev_prior = self.state_process_posterior_0.detach()
        self.m1y = self.state_process_posterior_0.detach()

    ###########################
    ### Initialize Monte Carlo ###
    ###########################

    def InitMonteCarlo(self):
        self.mc_idx = 0  # Indices for Monte Carlo

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):

        # Compute the 1-st moment of x based on model knowledge and without process noise
        self.state_process_prior_0[:, self.mc_idx] = torch.matmul(self.F, self.state_process_posterior_0[:, self.mc_idx])

        # Compute the 1-st moment of y based on model knowledge and without noise
        self.obs_process_0[:, self.mc_idx] = torch.matmul(self.H, self.state_process_prior_0[:, self.mc_idx])

        # Predict the 1-st moment of x
        self.m1x_prev_prior[:, self.mc_idx] = self.m1x_prior[:, self.mc_idx]
        self.m1x_prior[:, self.mc_idx] = torch.matmul(self.F, self.m1x_posterior[:, self.mc_idx])

        # Predict the 1-st moment of y
        self.m1y[:, self.mc_idx] = torch.matmul(self.H, self.m1x_prior[:, self.mc_idx])


    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y, cov):

        # Reshape and Normalize the difference in X prior
        # Featture 4: x_t|t - x_t|t-1
        #dm1x = self.m1x_prior - self.state_process_prior_0
        dm1x = self.m1x_posterior[:, self.mc_idx] - self.m1x_prev_prior[:, self.mc_idx]
        dm1x_reshape = torch.squeeze(dm1x)
        dm1x_norm = func.normalize(dm1x_reshape, p=2, dim=0, eps=1e-12, out=None)

        # Feature 2: yt - y_t+1|t
        dm1y = y - torch.squeeze(self.m1y[:,self.mc_idx])
        dm1y_norm = func.normalize(dm1y, p=2, dim=0, eps=1e-12, out=None)

        # new Feature: cov_t-1
        dm_cov_norm = func.normalize(cov, p=2, dim=0, eps=1e-12, out=None)

        # KGain Net Input
        KGainNet_in = torch.cat([dm1y_norm, dm1x_norm, dm_cov_norm], dim=0)

        # Kalman Gain Network Step
        KG,regularization = self.KGain_step(KGainNet_in)
        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.m, self.n))

        self.regularization = regularization

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y, cov):
        # Compute Priors
        self.step_prior()

        # Compute Kalman Gain
        self.step_KGain_est(y, cov)

        # Innovation
        y_obs = torch.unsqueeze(y, 1)
        dy = y_obs - self.m1y[:, self.mc_idx].unsqueeze(-1)

        # Compute the 1-st posterior moment
        INOV = torch.matmul(self.KGain, dy)
        self.m1x_posterior[:, self.mc_idx] = self.m1x_prior[:, self.mc_idx] + INOV.squeeze(-1)
        H = self.H
        K = self.KGain
        I = torch.eye(self.m)
        Ht = self.H_T
        Htilde = torch.linalg.inv((Ht @ H).cpu()).to(self.device)
        # HtildeHt = Htilde @ H.T
        # invI_HK = torch.linalg.inv(torch.eye(self.m) - H @ K)
        # HKR = H @ K @ self.R
        # HHtilde = H @ Htilde
        invKH_I = torch.linalg.inv((K @ H - torch.eye(self.m)).cpu()).to(self.device)

        cov_pred_byK_opt2 = - invKH_I @ K @ self.R @ H @ Htilde
        cov_post_byK_optA = (I - K @ H) @ cov_pred_byK_opt2
        # end compute cov


        # return
        return torch.squeeze(self.m1x_posterior[:, self.mc_idx-1]) ,self.regularization, torch.diagonal(cov_post_byK_optA)

    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, KGainNet_in):
        regularization = torch.empty(2)
        ###################
        ### Input Layer ###
        ###################
        L1_out, regularization[0] = self.KG_l1_con(KGainNet_in, self.KG_l1)
        ###########
        ### GRU ###
        ###########
        GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim).to(self.device,non_blocking = True)
        GRU_in[0, 0, :] = L1_out
        GRU_out, self.hn = self.rnn_GRU(GRU_in, self.hn)
        GRU_out_reshape = torch.reshape(GRU_out, (1, self.hidden_dim))


        L2_out,regularization[1] = self.KG_l2_con(GRU_out_reshape, self.KG_l2)
        return L2_out,regularization

    ###############
    ### Forward ###
    ###############
    def forward(self, yt, cov_t):
        yt = yt.to(self.device,non_blocking = True)
        cov_t = cov_t.to(self.device, non_blocking=True)
        return self.KNet_step(yt, cov_t)

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        self.hn = hidden.data
