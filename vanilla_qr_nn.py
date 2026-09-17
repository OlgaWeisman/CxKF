import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 use_bn=False, actv_type='relu', dropout=0.0):
        super().__init__()

        layers = [nn.Linear(in_features, out_features, bias=bias)]

        if use_bn:
            layers.append(nn.BatchNorm1d(out_features))

        if actv_type is not None:
            actv_type = actv_type.lower()
            if actv_type == 'relu':
                layers.append(nn.ReLU())
            elif actv_type == 'tanh':
                layers.append(nn.Tanh())
            elif actv_type == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif actv_type == 'leakyrelu':
                layers.append(nn.LeakyReLU())
            else:
                raise ValueError(f"Unsupported activation type: {actv_type}")

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class VanillaQRNN(nn.Module):
    def __init__(self,
                 input_size,
                 num_u,
                 architecture=1,
                 bias=True,
                 hidden_dimensions=[64, 64, 64],
                 use_bn=False,
                 actv_type='relu',
                 dropout=0.0,
                 H1=64,
                 HGRU=256,
                 H2=64,
                 nGRU=1):
        super().__init__()

        self.architecture = architecture
        self.num_u = num_u

        ####################################################
        # Architecture 1 (UNCHANGED)
        ####################################################
        if architecture == 1:

            self.fcs = nn.ModuleList()

            if len(hidden_dimensions) == 0:
                self.fcs.append(
                    LinearLayer(input_size, num_u, bias=bias,
                                use_bn=False, actv_type=None, dropout=0.0)
                )
            else:
                self.fcs.append(
                    LinearLayer(input_size, hidden_dimensions[0], bias=bias,
                                use_bn=use_bn, actv_type=actv_type, dropout=dropout)
                )

                for i in range(len(hidden_dimensions) - 1):
                    self.fcs.append(
                        LinearLayer(hidden_dimensions[i],
                                    hidden_dimensions[i + 1],
                                    bias=bias,
                                    use_bn=use_bn,
                                    actv_type=actv_type,
                                    dropout=dropout)
                    )

                self.fcs.append(
                    LinearLayer(hidden_dimensions[-1],
                                num_u,
                                bias=bias,
                                use_bn=False,
                                actv_type=None,
                                dropout=0.0)
                )

        ####################################################
        # Architecture 2 (NEW)
        ####################################################
        elif architecture == 2:

            self.hn = None

            self.input_layer = LinearLayer(
                input_size,
                H1,
                bias=bias,
                use_bn=False,
                actv_type="relu",
                dropout=dropout,
            )

            self.gru = nn.GRU(
                input_size=H1,
                hidden_size=HGRU,
                num_layers=nGRU,
                batch_first=False,
                dropout= dropout,
            )

            self.hidden_layer = LinearLayer(
                HGRU,
                H2,
                bias=bias,
                use_bn=False,
                actv_type="relu",
                dropout=0,
            )

            self.output_layer = LinearLayer(
                H2,
                num_u,
                bias=bias,
                use_bn=False,
                actv_type=None,
                dropout=0.0,
            )

        else:
            raise ValueError("architecture must be 1 or 2")

    def init_hidden(self, batch_size, device):
        if self.architecture == 2:
            self.hn = torch.zeros(
                self.gru.num_layers,
                batch_size,
                self.gru.hidden_size,
                device=device,
            )

    def forward(self, x):

        ####################################################
        # Architecture 1 (UNCHANGED)
        ####################################################
        if self.architecture == 1:
            for layer in self.fcs:
                x = layer(x)
            return x

        ####################################################
        # Architecture 2 (NEW)
        ####################################################
        # x shape: [batch_size, input_size]

        # Input layer
        x = self.input_layer(x)

        # [B,H1] -> [1,B,H1]
        gru_in = x.unsqueeze(0)

        # Initialize hidden state automatically
        if self.hn is None or self.hn.shape[1] != x.shape[0]:
            self.init_hidden(x.shape[0], x.device)
        # GRU
        gru_out, self.hn = self.gru(gru_in,self.hn)

        # [1,B,HGRU] -> [B,HGRU]
        gru_out = gru_out.squeeze(0)

        # Hidden layer
        x = self.hidden_layer(gru_out)

        # Output layer
        x = self.output_layer(x)

        return x

    def loss(self, y, x, u_list, tau_list):
        return multivariate_qr_loss(
            model=self,
            y=y,
            x=x,
            u_list=u_list,
            tau_list=tau_list
        )
    def loss_fixing(self, y, x, u_list, tau_list,q_gauss):
        return multivariate_qr_loss_fix(
            model=self,
            y=y,
            x=x,
            u_list=u_list,
            tau_list=tau_list,
            q_gauss=q_gauss
        )


def predict_y(model, x):
    """
    model input: x
    model output: [num_pts, num_u]
    """
    pred = model(x)
    return pred


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

def multivariate_qr_loss_fix(model, y, x, u_list, tau_list,q_gauss):
    """
    model output: [num_pts, num_u]
    y:            [num_pts, y_dim]
    u_list:       [num_u, y_dim]
    tau_list:     [num_u] or [num_pts, num_u]
    """
    Q_u_low = calc_y_u(u_list, q_gauss)

    pred = predict_y(model, x)    # [num_pts, num_u]
    pred = Q_u_low - pred # new part
    Y_u = calc_y_u(u_list, y)     # [num_pts, num_u]

    diff = Y_u - pred

    if tau_list.dim() == 1:
        tau_list = tau_list.unsqueeze(0)  # [1, num_u]

    mask = (tau_list - diff.le(0).float()).detach()
    pinball_loss = (mask * diff).mean()

    return pinball_loss