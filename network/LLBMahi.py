import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from network.utils import LinearFA, similarity_matrix


class LocalLossBlockMahi(nn.Module):
    """A module containing nn.Linear -> nn.BatchNorm1d -> nn.ReLU -> nn.Dropout
       The block can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.

    Args:
        num_in (int): Number of input features to linear layer.
        num_out (int): Number of output features from linear layer.
        num_classes (int): Number of classes (used in local prediction loss).
        first_layer (bool): True if this is the first layer in the network (used in local reconstruction loss).
        dropout (float): Dropout rate, if None, read from args.dropout.
        batchnorm (bool): True if to use batchnorm, if None, read from args.no_batch_norm.
    """

    def __init__(self, num_in, num_out, num_classes, first_layer=False, dropout=None, batchnorm=None, args=None):
        super(LocalLossBlockMahi, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.first_layer = first_layer
        self.dropout_p = args.dropout if dropout is None else dropout
        self.batchnorm = not args.no_batch_norm if batchnorm is None else batchnorm
        self.encoder = nn.Linear(num_in, num_out, bias=True)

        if not args.backprop and args.loss_unsup == 'recon':
            self.decoder_x = nn.Linear(num_out, num_in, bias=True)
        if not args.backprop and (args.loss_sup == 'pred' or args.loss_sup == 'predsim'):
            if args.bio:
                self.decoder_y = LinearFA(num_out, args.target_proj_size)
            else:
                self.decoder_y = nn.Linear(num_out, num_classes)
            self.decoder_y.weight.data.zero_()
        if not args.backprop and args.bio:
            self.proj_y = nn.Linear(num_classes, args.target_proj_size, bias=False)
        if not args.backprop and not args.bio and (
                args.loss_unsup == 'sim' or args.loss_sup == 'sim' or args.loss_sup == 'predsim'):
            self.linear_loss = nn.Linear(num_out, num_out, bias=False)
        if self.batchnorm:
            self.bn = torch.nn.BatchNorm1d(num_out)
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)
        if args.nonlin == 'relu':
            self.nonlin = nn.ReLU(inplace=True)
        elif args.nonlin == 'leakyrelu':
            self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
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

    def forward(self, x, y, y_onehot):
        # The linear transformation
        h = self.encoder(x)

        # Add batchnorm and nonlinearity
        if self.batchnorm:
            h = self.bn(h)
        h = self.nonlin(h)

        # Save return value and add dropout
        h_return = h
        if self.dropout_p > 0:
            h_return = self.dropout(h_return)

        # Calculate local loss and update weights
        if (self.training or not self.args.no_print_stats) and not self.args.backprop:
            # Calculate hidden layer similarity matrix
            if self.args.loss_unsup == 'sim' or self.args.loss_sup == 'sim' or self.args.loss_sup == 'predsim':
                if self.args.bio:
                    h_loss = h
                else:
                    h_loss = self.linear_loss(h)
                Rh = similarity_matrix(h_loss)

            # Calculate unsupervised loss
            if self.args.loss_unsup == 'sim':
                Rx = similarity_matrix(x).detach()
                loss_unsup = F.mse_loss(Rh, Rx)
            elif self.args.loss_unsup == 'recon' and not self.first_layer:
                x_hat = self.nonlin(self.decoder_x(h))
                loss_unsup = F.mse_loss(x_hat, x.detach())
            else:
                if self.args.cuda:
                    loss_unsup = torch.cuda.FloatTensor([0])
                else:
                    loss_unsup = torch.FloatTensor([0])

            # Calculate supervised loss
            if self.args.loss_sup == 'sim':
                if self.args.bio:
                    Ry = similarity_matrix(self.proj_y(y_onehot)).detach()
                else:
                    Ry = similarity_matrix(y_onehot).detach()
                loss_sup = F.mse_loss(Rh, Ry)
                if not self.args.no_print_stats:
                    self.loss_sim += loss_sup.item() * h.size(0)
                    self.examples += h.size(0)
            elif self.args.loss_sup == 'pred':
                y_hat_local = self.decoder_y(h.view(h.size(0), -1))
                if self.args.bio:
                    float_type = torch.cuda.FloatTensor if self.args.cuda else torch.FloatTensor
                    y_onehot_pred = self.proj_y(y_onehot).gt(0).type(float_type).detach()
                    loss_sup = F.binary_cross_entropy_with_logits(y_hat_local, y_onehot_pred)
                else:
                    loss_sup = F.cross_entropy(y_hat_local, y.detach())
                if not self.args.no_print_stats:
                    self.loss_pred += loss_sup.item() * h.size(0)
                    self.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
                    self.examples += h.size(0)
            elif self.args.loss_sup == 'predsim':
                y_hat_local = self.decoder_y(h.view(h.size(0), -1))
                if self.args.bio:
                    Ry = similarity_matrix(self.proj_y(y_onehot)).detach()
                    float_type = torch.cuda.FloatTensor if self.args.cuda else torch.FloatTensor
                    y_onehot_pred = self.proj_y(y_onehot).gt(0).type(float_type).detach()
                    loss_pred = (1 - self.args.beta) * F.binary_cross_entropy_with_logits(y_hat_local, y_onehot_pred)
                else:
                    Ry = similarity_matrix(y_onehot).detach()
                    loss_pred = (1 - self.args.beta) * F.cross_entropy(y_hat_local, y.detach())
                loss_sim = self.args.beta * F.mse_loss(Rh, Ry)
                loss_sup = loss_pred + loss_sim
                if not self.args.no_print_stats:
                    self.loss_pred += loss_pred.item() * h.size(0)
                    self.loss_sim += loss_sim.item() * h.size(0)
                    self.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
                    self.examples += h.size(0)

            # Combine unsupervised and supervised loss
            loss = self.args.alpha * loss_unsup + (1 - self.args.alpha) * loss_sup

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
