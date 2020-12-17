import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv_out_size(in_size, ks, strd):
    """ Calculate output size of a 1D convolutional layer
    Params
    ======
        in_size (int): input length
        ks (int): kernel size
        strd (int): stride of kernel over the input
    """
    return (in_size - ks + strd)//strd


class View(nn.Module):
    """ A simple View layer module for reshaping a tensor """
    def __init__(self, full_shape):
        super(View, self).__init__()
        self.shape = full_shape

    def forward(self, x):
        return x.view(self.shape)


class Network(nn.Module):
    def __init__(self, input_size, output_size, n_filt=(8, 16, 32), kernel_size=(9, 5, 3), stride=(2, 2, 1), fc_units=(128, 64, 32), actor=False):
        super(Network, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""

        self.layers = nn.ModuleList()  # container for layer objects

        # Convolutional layers
        out_size = input_size
        nf_old = 1
        for nf, ks, strd in zip(n_filt, kernel_size, stride):
            self.layers.append(nn.Conv1d(nf_old, nf, kernel_size=ks, stride=strd))  # conv1d layers with BN
            self.layers.append(nn.BatchNorm1d(nf))
            out_size = conv_out_size(out_size, ks, strd)
            nf_old = nf
        self.flat_size = n_filt[-1]*out_size  # calculate final flattened output size of conv layers

        # View layer for flattening
        self.layers.append(View((-1, self.flat_size)))

        # Feed-Forward (fully-connected) layers
        self.layers.append(nn.Linear(self.flat_size, fc_units[0]))     # first fc layer with BN
        for i in range(1, len(fc_units)):
            self.layers.append(nn.Linear(fc_units[i-1], fc_units[i]))  # middle fc layers with BN
        self.layers.append(nn.Linear(fc_units[-1], output_size))       # last fc layer

        self.nonlin = F.leaky_relu  # leaky_relu
        self.actor = actor

    def forward(self, x):
        x = x.unsqueeze(-2)  # add dimension for the convolutions
        x = self.nonlin(self.layers[0](x))  # first layer
        for layer in self.layers[1:-1]:
            x = self.nonlin(layer(x))  # middle layers
        # last layer
        if self.actor:
            return torch.tanh(self.layers[-1](x))
        else:
            return self.layers[-1](x)
