import autograd.numpy as np
from autograd import grad, jacobian
import torch,math
from torch import autograd

from parameters import m, n

def f(x, alpha=0.07, K=2, p=4):
    # return x + alpha * x * (1 - (x / K)**p)
    return torch.sin(x)#3*torch.sin(1.1*x + 0.1*math.pi)
def f_tylor(x):
   return x #(x - x**3 / 6)
def h(x):
    return x**2 #0.07*(0.01*x + 1)**2

    #return 10 * x / (10 + torch.abs(x))
# def f(x):
#    return torch.sin(x)
# def f(x, omega=1.0, kappa=0.3, dt=1.0):
#     # Pendulum-like angular dynamics with inlined wrapping to (-pi, pi]
#     phi = x + omega * dt - kappa * torch.sin(x) * dt
#     return torch.atan2(torch.sin(phi), torch.cos(phi))
# def f(x, lam=1.6, eps=0.3):
#     return 0.95*x + 0.25*torch.sin(x)
# def f(x, alpha=0.095, beta=5e-8, dt=1.0):
#     return x + alpha*x*dt + beta*(x**3)*dt
# def h(x):
#      return x ** 2
# favorit
# def f(x, alpha=0.08, K=100.0):
#     # x_{t+1} = x_t + alpha*x_t - (alpha/K^2)*x_t^3  → saturates at ±K
#     return x + alpha*x - (alpha/(K**2))*(x**3)
#
# def h(x):
#     # smooth, bounded measurement
#     return torch.tanh(x / 100)


def getJacobian(x, a):
    if m == 1 and n == 1:
        if (a == 'ObsAcc'):
            g = h
        elif (a == 'ModAcc'):
            g = f

        return autograd.functional.jacobian(g, x).unsqueeze(0)
    try:
        if (x.size()[1] == 1):
            y = torch.reshape((x.T), [x.size()[0]])
        else:
            y = x
    except:
        y = torch.reshape((x.T), [x.size()[0]])

    if (a == 'ObsAcc'):
        g = h
    elif (a == 'ModAcc'):
        g = f

    return autograd.functional.jacobian(g, y)
