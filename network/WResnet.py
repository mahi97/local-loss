import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from network.LLBConv import LocalLossBlockConv


class WideBasic(nn.Module):
    """ Used in WideResNet() """

    def __init__(self, in_planes, planes, dropout_rate, stride, num_classes, input_dim, dim_in_decoder, adapted, args):
        super(WideBasic, self).__init__()
        self.adapted = adapted
        self.args = args
        self.conv1 = LocalLossBlockConv(in_planes, planes, 3, 1, 1, num_classes, input_dim * stride, dim_in_decoder,
                                        dropout=None if self.adapted else 0, bias=True, pre_act=True, post_act=False, args=self.args)
        if not self.adapted:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = LocalLossBlockConv(planes, planes, 3, stride, 1, num_classes, input_dim, dim_in_decoder,
                                        dropout=None if self.adapted else 0, bias=True, pre_act=True, post_act=False, args=self.args)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )
            if args.optim == 'sgd':
                self.optimizer = optim.SGD(self.shortcut.parameters(), lr=0, weight_decay=args.weight_decay,
                                           momentum=args.momentum)
            elif args.optim == 'adam' or args.optim == 'amsgrad':
                self.optimizer = optim.Adam(self.shortcut.parameters(), lr=0, weight_decay=args.weight_decay,
                                            amsgrad=args.optim == 'amsgrad')

    def set_learning_rate(self, lr):
        self.lr = lr
        self.conv1.set_learning_rate(lr)
        self.conv2.set_learning_rate(lr)
        if len(self.shortcut) > 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def optim_zero_grad(self):
        self.conv1.optim_zero_grad()
        self.conv2.optim_zero_grad()
        if len(self.shortcut) > 0:
            self.optimizer.zero_grad()

    def optim_step(self):
        self.conv1.optim_step()
        self.conv2.optim_step()
        if len(self.shortcut) > 0:
            self.optimizer.step()

    def forward(self, input):
        x, y, y_onehot, loss_total = input
        out, loss = self.conv1(x, y, y_onehot)
        loss_total += loss
        if not self.adapted:
            out = self.dropout(out)
        out, loss = self.conv2(out, y, y_onehot, self.shortcut(x))
        loss_total += loss
        if not self.args.no_detach:
            if len(self.shortcut) > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        return out, y, y_onehot, loss_total


class WideResNet(nn.Module):
    """
    Wide residual network. The network can be trained by backprop or by locally generated error signal based on
    cross-entropy and/or similarity matching loss.
    """

    def __init__(self, depth, widen_factor, dropout_rate, num_classes, input_ch, input_dim, dim_in_decoder, args, adapted=False):
        super(WideResNet, self).__init__()
        self.adapted = adapted
        self.args = args
        self.dim_in_decoder = dim_in_decoder
        self.num_classes = num_classes
        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor

        print('| Wide-Resnet %dx%d %s' % (depth, k, 'adapted' if adapted else ''))
        if self.adapted:
            nStages = [16 * k, 16 * k, 32 * k, 64 * k]
        else:
            nStages = [16, 16 * k, 32 * k, 64 * k]
        self.in_planes = nStages[0]

        self.conv1 = LocalLossBlockConv(input_ch, nStages[0], 3, 1, 1, num_classes, 32, dim_in_decoder, dropout=0,
                                        bias=True, post_act=False, args=self.args)
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, 1, input_dim,  adapted)
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, 2, input_dim, adapted)
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, 2, input_dim // 2, adapted)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3] * (16 if self.adapted else 1), num_classes)
        if not self.args.backprop:
            self.linear.weight.data.zero_()

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, input_dim, adapted):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        stride_cum = 1
        for stride in strides:
            stride_cum *= stride
            layers.append(
                block(self.in_planes, planes, dropout_rate, stride, self.num_classes, input_dim // stride_cum,
                      self.dim_in_decoder, adapted, self.args))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def parameters(self, **kwargs):
        if not self.args.backprop:
            return self.linear.parameters()
        else:
            return super(WideResNet, self).parameters()

    def set_learning_rate(self, lr):
        self.conv1.set_learning_rate(lr)
        for layer in self.layer1:
            layer.set_learning_rate(lr)
        for layer in self.layer2:
            layer.set_learning_rate(lr)
        for layer in self.layer3:
            layer.set_learning_rate(lr)

    def optim_zero_grad(self):
        self.conv1.optim_zero_grad()
        for layer in self.layer1:
            layer.optim_zero_grad()
        for layer in self.layer2:
            layer.optim_zero_grad()
        for layer in self.layer3:
            layer.optim_zero_grad()

    def optim_step(self):
        self.conv1.optim_step()
        for layer in self.layer1:
            layer.optim_step()
        for layer in self.layer2:
            layer.optim_step()
        for layer in self.layer3:
            layer.optim_step()

    def forward(self, x, y, y_onehot):
        x, loss = self.conv1(x, y, y_onehot)
        x, _, _, loss = self.layer1((x, y, y_onehot, loss))
        x, _, _, loss = self.layer2((x, y, y_onehot, loss))
        x, _, _, loss = self.layer3((x, y, y_onehot, loss))
        x = F.relu(self.bn1(x))
        if self.adapted:
            x = F.max_pool2d(x, 2)
        else:
            x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x, loss
