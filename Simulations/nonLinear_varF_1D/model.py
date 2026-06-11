import autograd.numpy as np
from autograd import grad, jacobian
import torch
from torch import autograd

from parameters import  m, n

# def f(x, t):
#     dt = 0.1
#     a0 = 0.5
#     a1 = 0.1           # ~20% of a0 (mild seasonality)
#     omega = 2 * torch.pi / 200  # 200-step season
#     b = 0.05           # cubic saturation > 0 keeps things bounded
#     growth = a0 + a1 * torch.sin(torch.as_tensor(omega * t))
#     return x + dt * (growth * x - b * x ** 3)

# def f(x, t):
#     """
#     x_{k+1} = x_k + dt * ( -lam * x_k + A * sin(x_k + phi0 + omega * t) )
#     Minimal: no kwargs, constants inside.
#     """
#     dt = 0.02
#     lam = 0.10
#     A = 1.80
#     omega = 2 * torch.pi / 40   # ≈ 40-step period
#     phi0= torch.pi / 2
#
#     tT= torch.as_tensor(t, dtype=x.dtype, device=x.device)
#     phi = phi0 + omega * tT
#     return x + dt * (-lam * x + A * torch.sin(x + phi))
#
# def f(x, t):
#     """
#     Thermal RC with time-varying convection (fan/wind) + mild nonlinearity:
#       x_{k+1} = x_k + dt * [ beta(k) * (T_amb(k) - x_k) + beta2 * sin(x_k) ]
#       beta(k) = beta0 * (1 + 0.5 * sin(omega * k))
#
#     Minimal signature: f(x, t) — constants baked in.
#     Works for scalar or batched x.
#     """
#     # --- fixed constants (tweak as you like) ---
#     dt     = 0.1              # time step
#     beta0  = 1.0 / 60.0       # base thermal rate (~60 s time constant)
#     beta2  = 0.02             # mild nonlinear term
#     omega  = 2 * torch.pi / 200  # seasonal modulation (period ~200 steps)
#
#     T0     = 25.0             # ambient baseline (e.g., °C)
#     dT     = 0.0              # set >0 for slowly varying ambient (e.g., 2.0)
#
#     # --- cast scalars to match x's dtype/device ---
#     to_x = lambda v: torch.as_tensor(v, dtype=x.dtype, device=x.device)
#     tT    = to_x(t)
#     dtT   = to_x(dt)
#     beta0 = to_x(beta0)
#     beta2 = to_x(beta2)
#     omega = to_x(omega)
#     T0    = to_x(T0)
#     dT    = to_x(dT)
#
#     # time-varying coefficients
#     beta_k  = beta0 * (1.0 + 0.5 * torch.sin(omega * tT))
#     T_amb_k = T0 + dT * torch.sin(omega * tT)  # constant if dT == 0
#
#     # state update
#     return x + dtT * (beta_k * (T_amb_k - x) + beta2 * torch.sin(x))
def f(x,  t):
    """
    Time-varying generalized logistic:
      x_{k+1} = x_k + alpha(t) * x_k * (1 - (x_k / K(t))^p)

    Defaults match your constants (alpha=0.07, K=100, p=4) with mild seasonality.
    """
    # base params (your values)
    alpha0 = 0.07
    K0     = 100.0
    p      = 4.0

    # small seasonal mods (set to 0.0 to disable)
    eps_a  = 0.25                 # amplitude for alpha(t)
    eps_K  = 0.20                 # amplitude for K(t)
    om_a   = 2 * torch.pi / 200   # period ~200 steps
    om_K   = 2 * torch.pi / 150   # period ~150 steps

    tT = torch.as_tensor(t, dtype=x.dtype, device=x.device)

    # time-varying coefficients
    alpha_t = alpha0 * (1.0 + eps_a * torch.sin(om_a * tT))
    K_t     = K0     * (1.0 + eps_K * torch.cos(om_K * tT))
    K_t     = torch.clamp(K_t, min=1e-8)  # safety

    return x + alpha_t * x * (1.0 - (x / K_t)**p)
def h(x):
    return torch.tanh(x / 100)



def getJacobian(x, a, t=None):
    if m == 1 and n == 1:
        if (a == 'ObsAcc'):
            g = h
            return autograd.functional.jacobian(g, x).unsqueeze(0)
        elif (a == 'ModAcc'):
            g = f
            return torch.autograd.functional.jacobian(lambda x: g(x, t), x, vectorize=True).unsqueeze(0)

