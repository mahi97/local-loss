import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from network.LLBConv import LocalLossBlockConv


class BasicBlock(nn.Module):
    """ Used in ResNet() """
    expansion = 1

    def __init__(self, in_planes, planes, stride, num_classes, input_dim, dim_in_decoder, args):
        super(BasicBlock, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.stride = stride
        self.conv1 = LocalLossBlockConv(in_planes, planes, 3, stride, 1, num_classes, input_dim, dim_in_decoder, bias=False,
                                        pre_act=args.pre_act, post_act=not args.pre_act, args=args)
        self.conv2 = LocalLossBlockConv(planes, planes, 3, 1, 1, num_classes, input_dim, dim_in_decoder, bias=False,
                                        pre_act=args.pre_act, post_act=not args.pre_act, args=args)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, groups=1),
                nn.BatchNorm2d(self.expansion * planes)
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
        out, loss = self.conv2(out, y, y_onehot, self.shortcut(x))
        loss_total += loss
        if not self.args.no_detach:
            if len(self.shortcut) > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        return (out, y, y_onehot, loss_total)


class Bottleneck(nn.Module):
    """ Used in ResNet() """
    expansion = 4

    def __init__(self, in_planes, planes, stride, num_classes, input_dim, dim_in_decoder, args):
        super(Bottleneck, self).__init__()
        self.args = args
        self.conv1 = LocalLossBlockConv(in_planes, planes, 1, 1, 0, num_classes, input_dim,
                                        dim_in_decoder, bias=False, args=self.args)
        self.conv2 = LocalLossBlockConv(planes, planes, 3, stride, 1, num_classes, input_dim // stride,
                                        dim_in_decoder, bias=False, args=self.args)
        self.conv3 = LocalLossBlockConv(planes, self.expansion * planes, 1, 1, 0, num_classes, input_dim // stride,
                                        dim_in_decoder, bias=False, args=self.args)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
            if self.args.optim == 'sgd':
                self.optimizer = optim.SGD(self.shortcut.parameters(), lr=0, weight_decay=self.args.weight_decay,
                                           momentum=self.args.momentum)
            elif self.args.optim == 'adam' or self.args.optim == 'amsgrad':
                self.optimizer = optim.Adam(self.shortcut.parameters(), lr=0, weight_decay=self.args.weight_decay,
                                            amsgrad=self.args.optim == 'amsgrad')

    def set_learning_rate(self, lr):
        self.lr = lr
        self.conv1.set_learning_rate(lr)
        self.conv2.set_learning_rate(lr)
        self.conv3.set_learning_rate(lr)
        if len(self.shortcut) > 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def optim_zero_grad(self):
        self.conv1.optim_zero_grad()
        self.conv2.optim_zero_grad()
        self.conv3.optim_zero_grad()
        if len(self.shortcut) > 0:
            self.optimizer.zero_grad()

    def optim_step(self):
        self.conv1.optim_step()
        self.conv2.optim_step()
        self.conv3.optim_step()
        if len(self.shortcut) > 0:
            self.optimizer.step()

    def forward(self, input):
        x, y, y_onehot, loss_total = input
        out, loss = self.conv1(x, y, y_onehot)
        loss_total += loss
        out, loss = self.conv2(out, y, y_onehot)
        loss_total += loss
        out, loss = self.conv3(out, y, y_onehot, self.shortcut(x))
        loss_total += loss

        if not self.args.no_detach:
            if len(self.shortcut) > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        return out, y, y_onehot, loss_total


class ResNet(nn.Module):
    """
    Residual network.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
    """

    def __init__(self, block, num_blocks, num_classes, input_ch, feature_multiplyer, input_dim, dim_in_decoder, args):
        super(ResNet, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.dim_in_decoder = dim_in_decoder
        block = BasicBlock if block == 'basic' else Bottleneck
        self.in_planes = int(feature_multiplyer * 64)
        self.conv1 = LocalLossBlockConv(input_ch, int(feature_multiplyer * 64), 3, 1, 1, num_classes, input_dim,
                                        dim_in_decoder, bias=False, post_act=not self.args.pre_act, args=self.args)
        self.layer1 = self._make_layer(block, int(feature_multiplyer * 64), num_blocks[0], 1, input_dim)
        self.layer2 = self._make_layer(block, int(feature_multiplyer * 128), num_blocks[1], 2, input_dim)
        self.layer3 = self._make_layer(block, int(feature_multiplyer * 256), num_blocks[2], 2, input_dim // 2)
        self.layer4 = self._make_layer(block, int(feature_multiplyer * 512), num_blocks[3], 2, input_dim // 4)
        self.linear = nn.Linear(int(feature_multiplyer * 512 * block.expansion), num_classes)
        if not self.args.backprop:
            self.linear.weight.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride, input_dim):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        stride_cum = 1
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.num_classes, input_dim // stride_cum,
                                self.dim_in_decoder, self.args))
            stride_cum *= stride
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def parameters(self, **kwargs):
        if not self.args.backprop:
            return self.linear.parameters()
        else:
            return super(ResNet, self).parameters()

    def set_learning_rate(self, lr):
        self.conv1.set_learning_rate(lr)
        for layer in self.layer1:
            layer.set_learning_rate(lr)
        for layer in self.layer2:
            layer.set_learning_rate(lr)
        for layer in self.layer3:
            layer.set_learning_rate(lr)
        for layer in self.layer4:
            layer.set_learning_rate(lr)

    def optim_zero_grad(self):
        self.conv1.optim_zero_grad()
        for layer in self.layer1:
            layer.optim_zero_grad()
        for layer in self.layer2:
            layer.optim_zero_grad()
        for layer in self.layer3:
            layer.optim_zero_grad()
        for layer in self.layer4:
            layer.optim_zero_grad()

    def optim_step(self):
        self.conv1.optim_step()
        for layer in self.layer1:
            layer.optim_step()
        for layer in self.layer2:
            layer.optim_step()
        for layer in self.layer3:
            layer.optim_step()
        for layer in self.layer4:
            layer.optim_step()

    def forward(self, x, y, y_onehot):
        x, loss = self.conv1(x, y, y_onehot)
        x, _, _, loss = self.layer1((x, y, y_onehot, loss))
        x, _, _, loss = self.layer2((x, y, y_onehot, loss))
        x, _, _, loss = self.layer3((x, y, y_onehot, loss))
        x, _, _, loss = self.layer4((x, y, y_onehot, loss))
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x, loss
