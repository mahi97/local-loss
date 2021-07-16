import torch
import torch.nn as nn
import torch.nn.functional as F

from network.LLBLiner import LocalLossBlockLinear


class Net(nn.Module):
    '''
    A fully connected network.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.

    Args:
        num_layers (int): Number of hidden layers.
        num_hidden (int): Number of units in each hidden layer.
        input_dim (int): Feature map height/width for input.
        input_ch (int): Number of feature maps for input.
        num_classes (int): Number of classes (used in local prediction loss).
    '''

    def __init__(self, num_layers, num_hidden, input_dim, input_ch, num_classes, args):
        super(Net, self).__init__()
        self.args = args
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        reduce_factor = 1
        self.layers = nn.ModuleList(
            [LocalLossBlockLinear(input_dim * input_dim * input_ch, num_hidden, num_classes, first_layer=True,
                                  args=args)])
        self.layers.extend([LocalLossBlockLinear(int(num_hidden // (reduce_factor ** (i - 1))),
                                                 int(num_hidden // (reduce_factor ** i)), num_classes, args=args) for i
                            in
                            range(1, num_layers)])
        self.layer_out = nn.Linear(int(num_hidden // (reduce_factor ** (num_layers - 1))), num_classes)
        if not self.args.backprop:
            self.layer_out.weight.data.zero_()

    def parameters(self):
        if not self.args.backprop:
            return self.layer_out.parameters()
        else:
            return super(Net, self).parameters()

    def set_learning_rate(self, lr):
        for i, layer in enumerate(self.layers):
            layer.set_learning_rate(lr)

    def optim_zero_grad(self):
        for i, layer in enumerate(self.layers):
            layer.optim_zero_grad()

    def optim_step(self):
        for i, layer in enumerate(self.layers):
            layer.optim_step()

    def forward(self, x, y, y_onehot):
        x = x.view(x.size(0), -1)
        total_loss = 0.0
        for i, layer in enumerate(self.layers):
            x, loss = layer(x, y, y_onehot)
            total_loss += loss
        x = self.layer_out(x)

        return x, total_loss
