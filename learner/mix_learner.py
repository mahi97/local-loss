import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
from bisect import bisect_right
import os

from arg import get_args
from datasets import get_dataset, NClassRandomSampler

from network import *

from tqdm import tqdm
import wandb


def to_one_hot(y, n_dims=None):
    """ Take integer tensor y with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


class MixLearner:
    def __init__(self, model, args, num_classes, train_loader, test_loader):
        self.optimizer = None
        self.optimizer_mix = None
        self.model = model
        self.args = args
        self.num_classes = num_classes
        self.train_loader = train_loader
        self.test_loader = test_loader
        if args.optim == 'sgd':
            self.optimizer_mix = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
            self.optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        elif args.optim == 'adam' or args.optim == 'amsgrad':
            self.optimizer_mix = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.optim == 'amsgrad')
            self.optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.optim == 'amsgrad')
        else:
            print('Unknown optimizer')

    def train(self, epoch, lr):
        """ Train model on train set"""
        self.model.train()
        correct = 0
        loss_total_local = 0
        loss_total_global = 0

        # Add progress bar
        if self.args.progress_bar:
            pbar = tqdm(total=len(self.train_loader))

        # Clear layerwise statistics
        if not self.args.no_print_stats:
            for m in self.model.modules():
                if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                    m.clear_stats()

        # Loop train set
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            target_ = target
            target_onehot = to_one_hot(target, self.num_classes)
            if self.args.cuda:
                target_onehot = target_onehot.cuda()

            # Clear accumulated gradient
            self.optimizer.zero_grad()
            self.optimizer_mix.zero_grad()
            self.model.optim_zero_grad()

            output, mix_loss, true_loss, local_loss = self.model(data, target, target_onehot)
            loss_total_global += mix_loss * data.size(0)
            loss_total_local += local_loss.item() * data.size(0)

            # Backward pass and optimizer step
            # This is only for mixer to true value
            mix_loss.backward()
            self.optimizer_mix.step()

            true_loss.backward()
            self.optimizer.step()
            # If special option for no detaching is set, update weights also in hidden layers
            if self.args.no_detach and batch_idx % 10 == 0:
                self.model.optim_step()

            pred = output.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target_).cpu().sum()

            # Update progress bar
            if self.args.progress_bar:
                pbar.set_postfix(loss=mix_loss.item(), refresh=False)
                pbar.update()

        if self.args.progress_bar:
            pbar.close()

        # Format and print debug string
        loss_average_local = loss_total_local / len(self.train_loader.dataset)
        loss_average_global = loss_total_global / len(self.train_loader.dataset)
        error_percent = 100 - 100.0 * float(correct) / len(self.train_loader.dataset)
        wandb.log({"Train Loss Local": loss_average_local,
                   "Train Loss Global": loss_average_global,
                   "Train Error": error_percent})
        string_print = 'Train epoch={}, ' \
                       'lr={:.2e}, ' \
                       'loss_local={:.4f}, ' \
                       'loss_global={:.4f}, ' \
                       'error={:.3f}%, ' \
                       'mem={:.0f}MiB, ' \
                       'max_mem={:.0f}MiB\n'.format(
            epoch,
            lr,
            loss_average_local,
            loss_average_global,
            error_percent,
            torch.cuda.memory_allocated() / 1e6,
            torch.cuda.max_memory_allocated() / 1e6)
        if not self.args.no_print_stats:
            for m in self.model.modules():
                if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                    string_print += m.print_stats()
        print(string_print)

        return loss_average_local + loss_average_global, error_percent, string_print

    def test(self, epoch):
        """ Run model on test set """
        self.model.eval()
        test_loss = 0
        correct = 0

        # Clear layerwise statistics
        if not self.args.no_print_stats:
            for m in self.model.modules():
                if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                    m.clear_stats()

        # Loop test set
        for data, target in self.test_loader:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            target_ = target
            target_onehot = to_one_hot(target, self.num_classes)
            if self.args.cuda:
                target_onehot = target_onehot.cuda()

            with torch.no_grad():
                output, _, _, _ = self.model(data, target, target_onehot)
                test_loss += F.cross_entropy(output, target).item() * data.size(0)
            pred = output.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target_).cpu().sum()

        # Format and print debug string
        loss_average = test_loss / len(self.test_loader.dataset)
        if self.args.loss_sup == 'predsim' and not self.args.backprop:
            loss_average *= (1 - self.args.beta)
        error_percent = 100 - 100.0 * float(correct) / len(self.test_loader.dataset)
        string_print = 'Test loss_global={:.4f}, error={:.3f}%\n'.format(loss_average, error_percent)
        wandb.log({"Test Loss Global": loss_average, "Test Error": error_percent})
        if not self.args.no_print_stats:
            for m in self.model.modules():
                if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                    string_print += m.print_stats()
        print(string_print)

        return loss_average, error_percent, string_print
