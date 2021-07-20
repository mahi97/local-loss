import torch
import torch.nn as nn
import torch.nn.functional as F


def similarity_matrix(x, no_similarity_std=False):
    """ Calculate adjusted cosine similarity matrix of size x.size(0) x x.size(0). """
    if x.dim() == 4:
        if not no_similarity_std and x.size(1) > 3 and x.size(2) > 1:
            z = x.view(x.size(0), x.size(1), -1)
            x = z.std(dim=2)
        else:
            x = x.view(x.size(0), -1)
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc ** 2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1, 0)).clamp(-1, 1)
    return R


class LossSim(nn.Module):
    def __init__(self, num_classes, num_out, args, conv=False, avg_pool=None):
        super(LossSim, self).__init__()
        self.loss_fn = nn.Linear(num_out, num_out, bias=False)
        if conv:
            self.loss_fn = nn.Conv2d(num_out, num_out, 3, 1, padding=1, bias=False)
        self.proj_y = nn.Linear(num_classes, args.target_proj_size, bias=False)
        self.args = args
        self.avg_pool = avg_pool

    def forward(self, x, h, y_onehot):
        if self.args.bio:
            h_loss = self.avg_pool(h) if self.avg_pool else h
        else:
            h_loss = self.loss_fn(h)
        Rh = similarity_matrix(h_loss)
        Rx = similarity_matrix(x).detach()
        loss_unsup = F.mse_loss(Rh, Rx)
        if self.args.bio:
            Ry = similarity_matrix(self.proj_y(y_onehot)).detach()
        else:
            Ry = similarity_matrix(y_onehot).detach()
        loss_sup = F.mse_loss(Rh, Ry)
        # if not self.args.no_print_stats:
        #     self.loss_sim += loss_sup.item() * x.size(0)
        #     self.examples += x.size(0)
        return loss_unsup, loss_sup
