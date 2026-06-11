import torch
import torch.nn as nn


class SingleLayerPerceptronQR(nn.Module):
    def __init__(self, input_size, num_u, bias=False):
        super().__init__()
        self.num_u = num_u
        self.linear = nn.Linear(input_size, num_u, bias=bias)

    def forward(self, x):
        return self.linear(x)

    def loss(self, y, x, u_list, tau_list):
        return multivariate_qr_loss(
            model=self,
            y=y,
            x=x,
            u_list=u_list,
            tau_list=tau_list
        )


def predict_y(model, x):
    """
    model input: x
    model output: [num_pts, num_u]
    """
    return model(x)


def calc_y_u(u_list, y):
    """
    u_list: [num_u, y_dim]
    y:      [num_pts, y_dim]

    returns:
    Y_u:    [num_pts, num_u]
    """
    num_pts = y.shape[0]
    Y_u = torch.bmm(
        u_list.unsqueeze(0).repeat(num_pts, 1, 1),
        y.unsqueeze(-1)
    ).squeeze(-1)
    return Y_u


def multivariate_qr_loss(model, y, x, u_list, tau_list):
    """
    model output: [num_pts, num_u]
    y:            [num_pts, y_dim]
    u_list:       [num_u, y_dim]
    tau_list:     [num_u] or [num_pts, num_u]
    """
    pred = predict_y(model, x)    # [num_pts, num_u]
    Y_u = calc_y_u(u_list, y)     # [num_pts, num_u]

    diff = Y_u - pred

    if tau_list.dim() == 1:
        tau_list = tau_list.unsqueeze(0)  # [1, num_u]

    mask = (tau_list - diff.le(0).float()).detach()
    pinball_loss = (mask * diff).mean()

    return pinball_loss