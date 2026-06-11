"""# **Class: KalmanNet**"""

import torch
import torch.nn as nn
import torch.nn.functional as func


class KalmanNetNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()
        self.device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #OLGA
        self.quantiles = [0.05, 0.95]
        self.num_quantiles = len(self.quantiles)
        # # Define the new head parameters FIRST
        # self.new_head_in_shape = 1
        # self.new_head_hidden_size = 64
        # self.new_head_dropout = 0.5

        # # Now build the new head
        # self.build_q_head()
        # self.init_new_head_weights()

    #############
    ### Build ###
    #############
    def Build(self, ssModel):

        self.InitSystemDynamics(ssModel.F, ssModel.H, ssModel.Q, ssModel.R)

        # Number of neurons in the 1st hidden layer
        H1_KNet = (ssModel.m + ssModel.n) * (10) * 8

        # Number of neurons in the 2nd hidden layer
        H2_KNet = (ssModel.m * ssModel.n) * 1 * (4)

        self.InitKGainNet(H1_KNet, H2_KNet)

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitKGainNet(self, H1, H2):
        # Input Dimensions
        D_in = self.m + self.n  # x(t-1), y(t)

        # Output Dimensions
        D_out = self.m * self.n  # Kalman Gain

        ###################
        ### Input Layer ###
        ###################
        # Linear Layer
        self.KG_l1 = torch.nn.Linear(D_in, H1, bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu1 = torch.nn.ReLU()

        ###########
        ### GRU ###
        ###########
        # Input Dimension
        self.input_dim = H1
        # Hidden Dimension
        self.hidden_dim = (self.m * self.m + self.n * self.n) * 10
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
        self.hn = torch.randn(self.seq_len_hidden, self.batch_size, self.hidden_dim).to(self.device,non_blocking = True)

        # Iniatialize GRU Layer
        self.rnn_GRU = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers)

        ####################
        ### Quantile head ##
        ####################
        self.Q_l2 = torch.nn.Linear(self.hidden_dim, H2, bias=True)
        self.Q_relu2 = torch.nn.ReLU()
        self.Q_l3 = torch.nn.Linear(H2, self.num_quantiles)

        ####################
        ### Hidden Layer ###
        ####################
        self.KG_l2 = torch.nn.Linear(self.hidden_dim, H2, bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu2 = torch.nn.ReLU()

        ####################
        ### Output Layer ###
        ####################
        self.KG_l3 = torch.nn.Linear(H2, D_out, bias=True)

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, F, H, Q, R):
        # Set State Evolution Matrix
        self.F = F.to(self.device,non_blocking = True)
        self.F_T = torch.transpose(F, 0, 1)
        self.m = self.F.size()[0]

        # Set Observation Matrix
        self.H = H.to(self.device,non_blocking = True)
        self.H_T = torch.transpose(H, 0, 1).to(self.device,non_blocking = True)
        self.n = self.H.size()[0]
        self.Q = Q.to(self.device,non_blocking = True)
        self.R =R.to(self.device,non_blocking = True)
    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0):

        self.m1x_prior = M1_0.to(self.device,non_blocking = True)

        self.m1x_posterior = M1_0.to(self.device,non_blocking = True)

        self.state_process_posterior_0 = M1_0.to(self.device,non_blocking = True)

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):

        # Compute the 1-st moment of x based on model knowledge and without process noise
        self.state_process_prior_0 = torch.matmul(self.F, self.state_process_posterior_0)

        # Compute the 1-st moment of y based on model knowledge and without noise
        self.obs_process_0 = torch.matmul(self.H, self.state_process_prior_0)

        # Predict the 1-st moment of x
        self.m1x_prev_prior = self.m1x_prior
        self.m1x_prior = torch.matmul(self.F, self.m1x_posterior)

        # Predict the 1-st moment of y
        self.m1y = torch.matmul(self.H, self.m1x_prior)


    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):

        # Reshape and Normalize the difference in X prior
        # Featture 4: x_t|t - x_t|t-1
        #dm1x = self.m1x_prior - self.state_process_prior_0
        dm1x = self.m1x_posterior - self.m1x_prev_prior
        dm1x_reshape = torch.squeeze(dm1x)
        dm1x_norm = func.normalize(dm1x_reshape, p=2, dim=0, eps=1e-12, out=None)

        # Feature 2: yt - y_t+1|t
        dm1y = y - torch.squeeze(self.m1y)
        dm1y_norm = func.normalize(dm1y, p=2, dim=0, eps=1e-12, out=None)

        # KGain Net Input
        if dm1y_norm.dim() == 0:
            dm1y_norm = dm1y_norm.unsqueeze(0)
        if dm1x_norm.dim() == 0:
            dm1x_norm = dm1x_norm.unsqueeze(0)
        KGainNet_in = torch.cat([dm1y_norm, dm1x_norm], dim=0)

        # Kalman Gain Network Step
        KG, quantiles = self.KGain_step(KGainNet_in)
        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.m, self.n))
        self.quantiles = quantiles

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y):
        # Compute Priors
        self.step_prior()

        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Innovation
        y_obs = torch.unsqueeze(y, 1)
        dy = y_obs - self.m1y

        # Compute the 1-st posterior moment
        INOV = torch.matmul(self.KGain, dy)
        self.m1x_posterior = self.m1x_prior + INOV
        # # calculate covariance
        # H = self.H
        # K = self.KGain
        # I = torch.eye(self.m)
        # Ht = self.H_T
        # Htilde = torch.linalg.inv((Ht @ H).cpu()).to(self.device)
        # # HtildeHt = Htilde @ H.T
        # # invI_HK = torch.linalg.inv(torch.eye(self.m) - H @ K)
        # # HKR = H @ K @ self.R
        # # HHtilde = H @ Htilde
        # invKH_I = torch.linalg.inv((K @ H - torch.eye(self.m)).cpu()).to(self.device)
        #
        # cov_pred_byK_opt2 = - invKH_I @ K @ self.R @ H @ Htilde
        # self.cov_post_byK_optA = (I - K @ H) @ cov_pred_byK_opt2
        # return
        return torch.squeeze(self.m1x_posterior), self.quantiles

    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, KGainNet_in):

        ###################
        ### Input Layer ###
        ###################
        L1_out = self.KG_l1(KGainNet_in);
        La1_out = self.KG_relu1(L1_out);

        ###########
        ### GRU ###
        ###########
        GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim).to(self.device,non_blocking = True)
        GRU_in[0, 0, :] = La1_out
        GRU_out, self.hn = self.rnn_GRU(GRU_in, self.hn)
        GRU_out_reshape = torch.reshape(GRU_out, (1, self.hidden_dim))


        ####################
        ### Quantile Layer##
        ####################

        Q2_out = self.Q_l2(GRU_out_reshape)
        Qa2_out = self.Q_relu2(Q2_out)

        Q3_out = self.Q_l3(Qa2_out)

        ####################
        ### Hidden Layer ###
        ####################
        L2_out = self.KG_l2(GRU_out_reshape)
        La2_out = self.KG_relu2(L2_out)

        ####################
        ### Output Layer ###
        ####################
        L3_out = self.KG_l3(La2_out)
        return L3_out, Q3_out

    ###############
    ### Forward ###
    ###############
    def forward(self, yt):
        yt = yt.to(self.device,non_blocking = True)
        return self.KNet_step(yt)

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        self.hn = hidden.data

    #########################
    ###### New q head #######
    #########################

    # def build_q_head(self):
    #     self.q_head = nn.Sequential(
    #         nn.Linear(self.new_head_in_shape, self.new_head_hidden_size),
    #         nn.ReLU(),
    #         nn.Dropout(self.new_head_dropout),
    #         nn.Linear(self.new_head_hidden_size, self.new_head_hidden_size),
    #         nn.ReLU(),
    #         nn.Dropout(self.new_head_dropout),
    #         nn.Linear(self.new_head_hidden_size, self.num_quantiles),
    #     )

    # def init_new_head_weights(self):
    #     with torch.no_grad():
    #         for m in self.q_head:
    #             if isinstance(m, nn.Linear):
    #                 nn.init.xavier_uniform_(m.weight)  # or kaiming_uniform_
    #                 nn.init.constant_(m.bias, 0.)

    # def forward_q_head(self, x):
    #     # Coerce shape to (batch, in_features)
    #     if x.dim() == 0:  # scalar -> (1,1)
    #         x = x.view(1, 1)
    #     elif x.dim() == 1:  # (F,) -> (1,F)
    #         x = x.view(1, -1)
    #     else:
    #         # already batched -> leave as is
    #         pass
    #     ans = self.q_head(x)
    #     return torch.squeeze(ans)
    #
    # ####################
    # ### Forward both ###
    # ####################
    # def forward(self, yt):
    #     hat_x_t = self.forward_mse(yt)       # old path
    #     q = self.forward_q_head(hat_x_t)  # new head
    #     return hat_x_t, q