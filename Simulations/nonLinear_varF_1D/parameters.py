import torch
import math

#########################
### Design Parameters ###
#########################
variance = 0
delta_t_gen =  1e-5
delta_t = 0.02
delta_t_test = 0.01


m = n = 1
theta = 1 * 2 * math.pi/360
T = 100

m1x_0 = torch.ones(m, 1)
m2x_0 = torch.zeros(m,m)


#######################
### True Parameters ###
#######################

# Noise Parameters
sigma_q = 1
sigma_r = 1

# Noise Matrices
Q = (sigma_q**2) * torch.eye(n)
R = (sigma_r**2) * torch.eye(m)

########################
### Model Parameters ###
########################

# Noise Parameters
sigma_q = 1
sigma_r = 1

# Noise Matrices
Q_mod = (sigma_q**2) * torch.eye(n)
R_mod = (sigma_r**2) * torch.eye(m)
