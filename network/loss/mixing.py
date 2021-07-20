import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from network.linear import LinearFA


class LossMix(nn.Module):
    def __init__(self, num_out, args, avg_pool=None):
        super(LossMix, self).__init__()
        self.args = args
        self.avg_pool = avg_pool
        # Set up network layers
        self.fc1 = nn.Linear(num_out, 128)
        # self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        if args.nonlin == 'relu':
            self.nonlin = nn.ReLU(inplace=True)
        elif args.nonlin == 'leakyrelu':
            self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, h):
        h_loss = h
        if self.avg_pool:
            h_loss = self.avg_pool(h)

        h_loss = self.nonlin(self.fc1(h.view(h_loss.size(0), -1)))
        # h_loss = self.nonlin(self.fc2(h_loss))
        q = self.fc3(h_loss)
        return torch.mean(q)
