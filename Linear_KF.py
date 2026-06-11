"""# **Class: Kalman Filter**
Theoretical Linear Kalman
"""
import torch
import numpy as np

class KalmanFilter:

    def __init__(self, SystemModel):
        self.F = SystemModel.F;
        self.F_T = torch.transpose(self.F, 0, 1);
        self.m = SystemModel.m

        self.Q = SystemModel.Q;

        self.H = SystemModel.H;
        self.H_T = torch.transpose(self.H, 0, 1);
        self.n = SystemModel.n

        self.R = SystemModel.R;

        self.T = SystemModel.T;
        self.T_test = SystemModel.T_test;
        # for change F
        self.t = 0
   
    # Predict

    def Predict(self):
        # Predict the 1-st moment of x
        self.m1x_prior = torch.matmul(self.F, self.m1x_posterior);

        # Predict the 2-nd moment of x
        self.m2x_prior = torch.matmul(self.F, self.m2x_posterior);
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F_T).to(self.Q.device) + self.Q;

        # Predict the 1-st moment of y
        self.m1y = torch.matmul(self.H, self.m1x_prior).to(self.Q.device);

        # Predict the 2-nd moment of y
        self.m2y = torch.matmul(self.H.to(self.Q.device), self.m2x_prior);
        self.m2y = torch.matmul(self.m2y.to(self.Q.device), self.H_T.to(self.Q.device)).to(self.Q.device) + self.R.to(self.Q.device);

    # Compute the Kalman Gain
    def KGain(self):
        self.KG = torch.matmul(self.m2x_prior, self.H_T);
        device = self.m2y.device  # 'cuda:0'

        m2y_cpu = self.m2y.to("cpu")
        KG_cpu = self.KG.to("cpu")

        inv_m2y = torch.inverse(m2y_cpu)  # or torch.linalg.inv(m2y_cpu)
        KG_cpu = torch.matmul(KG_cpu, inv_m2y)

        self.KG = KG_cpu.to(device)  # back to GPU

    # Innovation
    def Innovation(self, y):
        self.dy = y.to(self.m1y.device) - self.m1y;

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.matmul(self.KG, self.dy);

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.matmul(self.m2y, torch.transpose(self.KG, 0, 1))
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG, self.m2x_posterior)

    def Update(self, y):
        self.Predict();
        self.KGain();
        self.Innovation(y);
        self.Correct();

        return self.m1x_posterior,self.m2x_posterior;

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

        #########################

    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, y, T):
        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, T])
        self.sigma = torch.empty(size=[self.m, self.m, T])

        self.m1x_posterior = self.m1x_0
        self.m2x_posterior = self.m2x_0
        for t in range(0, T):
            if self.m == 1 & self.n == 1:
                yt = y[:,t]
            else:
                yt = torch.unsqueeze(y[:, t], 1);
            xt,sigmat = self.Update(yt);
            if self.m == 1 & self.n == 1:
                self.x[:, t] = torch.squeeze(xt)
                self.sigma[:, :, t] = torch.squeeze(sigmat)
            else:
                self.x[:, t] = torch.squeeze(xt)
                self.sigma[:, :, t] = torch.squeeze(sigmat)

