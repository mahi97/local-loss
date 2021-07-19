"""
Code adopted from here:
https://github.com/oxwhirl/pymarl/blob/master/src/modules/mixers/vdn.py
"""

import torch
import torch.nn as nn


class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs, batch):
        return torch.sum(agent_qs, dim=2, keepdim=True)
