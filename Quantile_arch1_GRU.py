"""# **Class: KalmanNet**"""

import torch
import torch.nn as nn
import torch.nn.functional as func


class QuantileNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(
        self,
        new_head_in_shape,          # <- number of input features to the head
        quantiles=(0.05, 0.95),
        new_head_hidden_size=64,
        new_head_dropout=0.5,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()

        self.device = device
        self.quantiles = list(quantiles)
        self.num_quantiles = len(self.quantiles)

        self.new_head_in_shape = new_head_in_shape
        self.new_head_hidden_size = new_head_hidden_size
        self.new_head_dropout = new_head_dropout

        self.build_q_head()

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        # new head
        weight = next(self.parameters()).data
        hidden = weight.new(1, 1, self.new_head_hidden_size).zero_()
        self.hn1 = hidden.data
    #########################
    ###### New q head #######
    #########################

    def build_q_head(self):
        # Initialize a Tensor for Hidden State
        hidden_dim = 64
        n_layers = 1

        self.fc1 = nn.Linear(self.new_head_in_shape, self.new_head_hidden_size)
        self.act1 = nn.ReLU()
        self.hn1 = torch.randn(n_layers, 1, self.new_head_hidden_size)
        # default GRU: expects (seq_len, batch, input_dim)
        self.gru = nn.GRU(input_size=self.new_head_hidden_size,
                          hidden_size=self.new_head_hidden_size,
                          num_layers=n_layers)

        self.fc2 = nn.Linear(self.new_head_hidden_size, self.new_head_hidden_size)
        self.act2 = nn.ReLU()

        self.fc_out = nn.Linear(self.new_head_hidden_size, self.num_quantiles)



    #def forward(self, x, cov):
    def forward(self, *args):
        x = args[0]
        if len(args) == 2:
            cov = args[1]
            input = torch.cat([x.view(-1), cov.view(-1)], dim=0)
        elif len(args) == 4:
            delta_x = args[1]
            delta_y = args[2]
            delta_z = args[3]
            input = torch.cat([x.view(-1), delta_x, delta_y, delta_z], dim=0)
        # ans = self.q_head(x)
        # return torch.squeeze(ans)
        # ----- Feedforward layer -----
        input = self.fc1(input)
        input = self.act1(input)

        # ----- Move into GRU format -----
        # # from (B, H) -> (1, B, H)  => seq_len = 1
        # input = input.unsqueeze(0)
        GRU_in = torch.empty(1, 1, self.new_head_hidden_size).to(self.device,non_blocking = True)
        GRU_in[0, 0, :] = input
        # GRU returns: (output, h_n)
        gru_out, self.hn1 = self.gru(GRU_in, self.hn1)    # output shape: (1, B, hidden_dim)

        # Take last step: (B, hidden_dim)
        input = gru_out[-1, :, :]

        # ----- More feedforward -----
        input = self.fc2(input)
        input = self.act2(input)

        # Final quantiles
        q = self.fc_out(input)          # (B, num_quantiles)
        return q
