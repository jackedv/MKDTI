import torch as t
from torch import nn


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, target, prediction, drug_lap, target_lap, alpha1, alpha2, sizes):
        loss_ls = t.norm((target - prediction), p='fro') ** 2

        drug_reg = t.trace(t.mm(t.mm(alpha1.T, drug_lap), alpha1))
        target_reg = t.trace(t.mm(t.mm(alpha2.T, target_lap), alpha2))
        graph_reg = sizes.lambda1 * drug_reg + sizes.lambda2 * target_reg

        loss_sum = loss_ls + graph_reg

        return loss_sum.sum()