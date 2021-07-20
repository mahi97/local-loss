import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from network.loss import LossSim, LossRecon, LossPred
from network.linear import LinearFA


class LocalLossBlockMahi(nn.Module):
    """A module containing nn.Linear -> nn.BatchNorm1d -> nn.ReLU -> nn.Dropout
       The block can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.

    Args:
        num_in (int): Number of input features to linear layer.
        num_out (int): Number of output features from linear layer.
        num_classes (int): Number of classes (used in local prediction loss).
        first_layer (bool): True if this is the first layer in the network (used in local reconstruction loss).
        dropout_p (float): Dropout rate, if None, read from args.dropout.
        batchnorm (bool): True if to use batchnorm, if None, read from args.no_batch_norm.
    """

    def __init__(self, num_in, num_out, num_classes, first_layer=False, dropout_p=None, batchnorm=None, args=None):
        super(LocalLossBlockMahi, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.sim = LossSim(num_classes, num_out, args)
        self.pred = LossPred(num_classes, num_out, args)
        self.recon = LossRecon(num_in, num_out, args, first_layer)

        self.dropout_p = args.dropout if dropout_p is None else dropout_p
        self.batchnorm = not args.no_batch_norm if batchnorm is None else batchnorm
        self.encoder = nn.Linear(num_in, num_out, bias=True)

        if self.batchnorm:
            self.bn = torch.nn.BatchNorm1d(num_out)
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)

        if self.dropout_p > 0:
            self.dropout = torch.nn.Dropout(p=self.dropout_p, inplace=False)

        if args.optim == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=0, weight_decay=args.weight_decay, momentum=args.momentum)
        elif args.optim == 'adam' or args.optim == 'amsgrad':
            self.optimizer = optim.Adam(self.parameters(), lr=0, weight_decay=args.weight_decay,
                                        amsgrad=args.optim == 'amsgrad')

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

    def _mahi_loss(self, h):
        return h

    def forward(self, x, y, y_onehot):
        # The linear transformation
        h = self.encoder(x)

        # Add batchnorm and nonlinearity
        h = self.bn(h) if self.batchnorm else h
        h = self.nonlin(h)

        # Save return value and add dropout
        h_return = h
        h_return = self.dropout(h_return) if self.dropout_p > 0 else h_return

        # Calculate local loss and update weights
        if (self.training or not self.args.no_print_stats) and not self.args.backprop:

            loss_recon = self.recon(x)
            loss_sim_u, loss_sim_s = self.sim(x, y_onehot)
            loss_pred = self.pred(x, y, y_onehot)
            loss_mahi = 0.0
            # Combine unsupervised and supervised loss
            loss = sum(map(lambda _x: _x[0] * _x[1], zip(self.args.abcde, [loss_recon, loss_sim_u, loss_sim_s, loss_pred, loss_mahi])))

            # Single-step back-propagation
            if self.training:
                loss.backward(retain_graph=self.args.no_detach)

            # Update weights in this layer and detach computational graph
            if self.training and not self.args.no_detach:
                self.optimizer.step()
                self.optimizer.zero_grad()
                h_return.detach_()

            loss = loss.item()
        else:
            loss = 0.0

        return h_return, loss
