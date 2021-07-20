import torch
import torch.nn as nn
import torch.nn.functional as F


class LossRecon(nn.Module):
    def __init__(self, num_in, num_out, args, first_layer=False,
                 conv=False, kernel_size=None, stride=None, padding=None, bias=None):
        super(LossRecon, self).__init__()
        self.decoder_x = nn.Linear(num_out, num_in, bias=True)
        if conv:
            self.bias = True if bias is None else bias
            self.decoder_x = nn.ConvTranspose2d(num_out, num_in, kernel_size, stride=stride, padding=padding, bias=bias)
        self.nonlin = None
        self.args = args
        self.first_layer = first_layer
        if args.nonlin == 'relu':
            self.nonlin = nn.ReLU(inplace=True)
        elif args.nonlin == 'leakyrelu':
            self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        if not self.first_layer:
            x_hat = self.nonlin(self.decoder_x(x))
            loss_unsup = F.mse_loss(x_hat, x.detach())
        else:
            if self.args.cuda:
                loss_unsup = torch.cuda.FloatTensor([0])
            else:
                loss_unsup = torch.FloatTensor([0])
        return loss_unsup
