import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from network.linear import LinearFA
from network.loss import LossSim, LossPred, LossRecon
import math


def get_avg_pool_and_dim_in_decoder(dim_out, num_out, _dim_in_decoder):
    # Resolve average-pooling kernel size in order for flattened dim to match self.args.dim_in_decoder
    avg_pool = None
    ks_h, ks_w = 1, 1
    dim_out_h, dim_out_w = dim_out, dim_out
    dim_in_decoder = num_out * dim_out_h * dim_out_w
    while dim_in_decoder > _dim_in_decoder and ks_h < dim_out:
        ks_h *= 2
        dim_out_h = math.ceil(dim_out / ks_h)
        dim_in_decoder = num_out * dim_out_h * dim_out_w
        if dim_in_decoder > _dim_in_decoder:
            ks_w *= 2
            dim_out_w = math.ceil(dim_out / ks_w)
            dim_in_decoder = num_out * dim_out_h * dim_out_w
    if ks_h > 1 or ks_w > 1:
        pad_h = (ks_h * (dim_out_h - dim_out // ks_h)) // 2
        pad_w = (ks_w * (dim_out_w - dim_out // ks_w)) // 2
        avg_pool = nn.AvgPool2d((ks_h, ks_w), padding=(pad_h, pad_w))

    return avg_pool, dim_in_decoder


class LocalLossBlockConv(nn.Module):
    '''
    A block containing nn.Conv2d -> nn.BatchNorm2d -> nn.ReLU -> nn.Dropou2d
    The block can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.

    Args:
        ch_in (int): Number of input features maps.
        ch_out (int): Number of output features maps.
        kernel_size (int): Kernel size in Conv2d.
        stride (int): Stride in Conv2d.
        padding (int): Padding in Conv2d.
        num_classes (int): Number of classes (used in local prediction loss).
        dim_out (int): Feature map height/width for input (and output).
        first_layer (bool): True if this is the first layer in the network (used in local reconstruction loss).
        dropout (float): Dropout rate, if None, read from Args.dropout.
        bias (bool): True if to use trainable bias.
        pre_act (bool): True if to apply layer order nn.BatchNorm2d -> nn.ReLU -> nn.Dropou2d -> nn.Conv2d (used for PreActResNet).
        post_act (bool): True if to apply layer order nn.Conv2d -> nn.BatchNorm2d -> nn.ReLU -> nn.Dropou2d.
    '''

    def __init__(self, ch_in, ch_out, kernel_size, stride, padding, num_classes, dim_out, first_layer=False,
                 dropout=None, bias=None, pre_act=False, post_act=True, args=None):
        super(LocalLossBlockConv, self).__init__()
        self.args = args
        avg_pool, dim_in_decoder = get_avg_pool_and_dim_in_decoder(dim_out, ch_out, args.dim_in_decoder)
        self.sim = LossSim(num_classes, ch_out, args, True, avg_pool)
        self.pred = LossPred(num_classes, dim_in_decoder, args, avg_pool)
        self.recon = LossRecon(ch_in, ch_out, args, first_layer, True, kernel_size, stride, padding, bias)
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.num_classes = num_classes
        self.first_layer = first_layer
        self.dropout_p = self.args.dropout if dropout is None else dropout
        self.pre_act = pre_act
        self.post_act = post_act
        self.encoder = nn.Conv2d(ch_in, ch_out, kernel_size, stride=stride, padding=padding, bias=bias)
        if args.nonlin == 'relu':
            self.nonlin = nn.ReLU(inplace=True)
        elif args.nonlin == 'leakyrelu':
            self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        if not self.args.no_batch_norm:
            if pre_act:
                self.bn_pre = torch.nn.BatchNorm2d(ch_in)
            if not (pre_act and self.args.backprop):
                self.bn = torch.nn.BatchNorm2d(ch_out)
                nn.init.constant_(self.bn.weight, 1)
                nn.init.constant_(self.bn.bias, 0)

        if self.dropout_p > 0:
            self.dropout = torch.nn.Dropout2d(p=self.dropout_p, inplace=False)
        if self.args.optim == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=0, weight_decay=self.args.weight_decay,
                                       momentum=self.args.momentum)
        elif self.args.optim == 'adam' or self.args.optim == 'amsgrad':
            self.optimizer = optim.Adam(self.parameters(), lr=0, weight_decay=self.args.weight_decay,
                                        amsgrad=self.args.optim == 'amsgrad')

        self.clear_stats()

    def clear_stats(self):
        if not self.args.no_print_stats:
            self.loss_sim = 0.0
            self.loss_pred = 0.0
            self.correct = 0
            self.examples = 0

    def print_stats(self):
        if not self.args.backprop:
            stats = '{}, loss_sim={:.4f}, loss_pred={:.4f}, error={:.3f}%, num_examples={}\n'.format(
                self.encoder,
                self.loss_sim / self.examples,
                self.loss_pred / self.examples,
                100.0 * float(self.examples - self.correct) / self.examples,
                self.examples)
            return stats
        else:
            return ''

    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def optim_zero_grad(self):
        self.optimizer.zero_grad()

    def optim_step(self):
        self.optimizer.step()

    def forward(self, x, y, y_onehot, x_shortcut=None):
        # If pre-activation, apply batchnorm->nonlin->dropout
        if self.pre_act:
            if not self.args.no_batch_norm:
                x = self.bn_pre(x)
            x = self.nonlin(x)
            if self.dropout_p > 0:
                x = self.dropout(x)

        # The convolutional transformation
        h = self.encoder(x)

        # If post-activation, apply batchnorm
        if self.post_act and not self.args.no_batch_norm:
            h = self.bn(h)

        # Add shortcut branch (used in residual networks)
        if x_shortcut is not None:
            h = h + x_shortcut

        # If post-activation, add nonlinearity
        if self.post_act:
            h = self.nonlin(h)

        # Save return value and add dropout
        h_return = h
        if self.post_act and self.dropout_p > 0:
            h_return = self.dropout(h_return)

        # Calculate local loss and update weights
        if (not self.args.no_print_stats or self.training) and not self.args.backprop:
            # Add batchnorm and nonlinearity if not done already
            if not self.post_act:
                if not self.args.no_batch_norm:
                    h = self.bn(h)
                h = self.nonlin(h)

            # Calculate hidden feature similarity matrix
            # loss_recon = self.recon(h)
            # loss_sim_u, loss_sim_s = self.sim(h, y_onehot)
            loss_pred = self.pred(x, y, y_onehot)
            # loss_mahi = 0.0

            loss = loss_pred #sum(map(lambda _x: _x[0] * _x[1], zip(self.args.abcde, [loss_recon, loss_sim_u, loss_sim_s, loss_pred])))

            # Single-step back-propagation
            if self.training:
                loss.backward(retain_graph=self.args.no_detach)

            # Update weights in this layer and detatch computational graph
            if self.training and not self.args.no_detach:
                self.optimizer.step()
                self.optimizer.zero_grad()
                h_return.detach_()

            loss = loss.item()
        else:
            loss = 0.0

        return h_return, loss
