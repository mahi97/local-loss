import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from network.linear import LinearFA


class LossPred(nn.Module):
    def __init__(self, num_classes, num_out, args, avg_pool=None):
        super(LossPred, self).__init__()
        self.args = args
        self.avg_pool = avg_pool
        self.decoder_y = nn.Linear(num_out, num_classes)
        if args.bio:
            self.decoder_y = LinearFA(num_out, args.target_proj_size, args)
            self.proj_y = nn.Linear(num_classes, args.target_proj_size, bias=False)
        self.decoder_y.weight.data.zero_()

    def forward(self, h, y, y_onehot):
        if self.avg_pool:
            h = self.avg_pool(h)
        y_hat_local = self.decoder_y(h.view(h.size(0), -1))
        if self.args.bio:
            float_type = torch.cuda.FloatTensor if self.args.cuda else torch.FloatTensor
            y_onehot_pred = self.proj_y(y_onehot).gt(0).type(float_type).detach()
            loss_sup = F.binary_cross_entropy_with_logits(y_hat_local, y_onehot_pred)
        else:
            loss_sup = F.cross_entropy(y_hat_local, y.detach())
        # if not self.args.no_print_stats:
        #     self.loss_pred += loss_sup.item() * x.size(0)
        #     self.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
        #     self.examples += x.size(0)
        return loss_sup
