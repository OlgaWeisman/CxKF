import torch
import torch.nn as nn
class ConcreteDropout(nn.Module):
    def __init__(self, weight_regularizer=1e-6,
                 dropout_regularizer=1e-6, init_min=0.01, init_max=0.3, device: str ='cpu'):

        super(ConcreteDropout, self).__init__()
        self.device = device
        # self.device = torch.device('cpu') #OLGA
        # # Device
        # if str(device.) == "cuda":
        #     self.device = torch.device('cuda')
        #     torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # else:
        #     self.device = torch.device('cpu')
        #     torch.set_default_tensor_type('torch.FloatTensor')

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        init_min = torch.log(torch.tensor(init_min)).to(self.device) - torch.log(torch.tensor(1. - init_min)).to(self.device)
        init_max = torch.log(torch.tensor(init_max)).to(self.device) - torch.log(torch.tensor(1. - init_max)).to(self.device)

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))

    def forward(self, x, layer ,currSeed=None):

        # If currSeed is provided, set the random seed
        if currSeed is not None:
            torch.manual_seed(currSeed)
            torch.cuda.manual_seed_all(currSeed)  # Ensures reproducibility across GPUs
        p = torch.sigmoid(self.p_logit)

        out = layer(self._concrete_dropout(x, p,currSeed))

        # sum_of_square = 0
        # for param in layer.parameters():
        #     sum_of_square += torch.sum(torch.pow(param, 2)).to(self.device)
        #
        # weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)

        # Calculate weight regularization more efficiently
        sum_of_square = sum(param.pow(2).sum() for param in layer.parameters())
        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)

        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)

        input_dimensionality = x[0].numel()  # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality

        regularization = weights_regularizer + dropout_regularizer # TODO Plus Or Minus ?
        return out,regularization

    def _concrete_dropout(self, x, p, currSeed):
        if currSeed is not None:
            torch.manual_seed(currSeed)
            torch.cuda.manual_seed_all(currSeed)  # Ensures reproducibility across GPUs

        eps = 1e-7
        temp = 0.1

        # # Ensure x and all tensors are on the correct device
        # x = x.to(self.device)
        # p = p.to(self.device) if isinstance(p, torch.Tensor) else torch.tensor(p, device=self.device)

        unif_noise = torch.rand_like(x).to(self.device)

        drop_prob = (torch.log(p + eps)
                     - torch.log(1 - p + eps)
                     + torch.log(unif_noise + eps)
                     - torch.log(1 - unif_noise + eps))

        drop_prob = torch.sigmoid(drop_prob / temp)

        random_tensor = 1 - drop_prob
        retain_prob = 1 - p

        return torch.mul(x, random_tensor).div_(retain_prob)