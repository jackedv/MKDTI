import torch as t
from torch import nn
from torch_geometric.nn import conv
from torch_geometric.nn import GATConv
from utils import *

import math

import torch
import torch.nn.functional as F

from cogdl.utils import edge_softmax, get_activation
from cogdl.utils.spmm_utils import MultiHeadSpMM

class Model(nn.Module):
    def __init__(self, sizes, drug_sim, target_sim):
        super(Model, self).__init__()
        np.random.seed(sizes.seed)
        t.manual_seed(sizes.seed)
        self.drug_size = sizes.drug_size
        self.target_size = sizes.target_size
        self.F1 = sizes.F1
        self.F2 = sizes.F2
        self.F3 = sizes.F3
        self.F4 = sizes.F4
        self.heads = sizes.heads
        self.seed = sizes.seed
        self.h1_gamma = sizes.h1_gamma
        self.h2_gamma = sizes.h2_gamma
        self.h3_gamma = sizes.h3_gamma
        self.h4_gamma = sizes.h4_gamma
        self.lambda1 = sizes.lambda1
        self.lambda2 = sizes.lambda2

        self.kernel_len = 5
        self.drug_ps = t.ones(self.kernel_len) / self.kernel_len
        self.target_ps = t.ones(self.kernel_len) / self.kernel_len
        self.drug_ps = t.tensor([1,0.5,0.333,0.25,0.2])
        self.mic_ps = t.tensor([1,0.5,0.333,0.25,0.2])

        self.drug_sim = t.DoubleTensor(drug_sim)
        self.target_sim = t.DoubleTensor(target_sim)

        self.gat_1 = conv.GATConv(self.drug_size + self.target_size, self.F1, self.heads[0])
        self.gat_2 = conv.GATConv(self.F1*self.heads[0], self.F2, self.heads[1])
        self.gat_3 = conv.GATConv(self.F2*self.heads[1], self.F3, self.heads[2])
        self.gat_4 = conv.GATConv(self.F3*self.heads[2], self.F4, self.heads[3])

        self.alpha1 = t.randn(self.drug_size, self.target_size).double()
        self.alpha2 = t.randn(self.target_size, self.drug_size).double()

        self.drug_l = []
        self.target_l = []

        self.drug_k = []
        self.target_k = []

    def forward(self, input):
        t.manual_seed(self.seed)
        x = input['feature']
        adj = input['Adj']
        drugs_kernels = []
        target_kernels = []

        H1 = t.relu(self.gat_1(x, adj['edge_index'], adj['data'][adj['edge_index'][0], adj['edge_index'][1]])) 
        drugs_kernels.append(t.DoubleTensor(getGipKernel(H1[:self.drug_size].clone(), 0, self.h1_gamma, True).double())) 
        target_kernels.append(t.DoubleTensor(getGipKernel(H1[self.drug_size:].clone(), 0, self.h1_gamma, True).double())) 

        H2 = t.relu(self.gat_2(H1, adj['edge_index'], adj['data'][adj['edge_index'][0], adj['edge_index'][1]]))
        drugs_kernels.append(t.DoubleTensor(getGipKernel(H2[:self.drug_size].clone(), 0, self.h2_gamma, True).double()))
        target_kernels.append(t.DoubleTensor(getGipKernel(H2[self.drug_size:].clone(), 0, self.h2_gamma, True).double()))

        H3 = t.relu(self.gat_3(H2, adj['edge_index'], adj['data'][adj['edge_index'][0], adj['edge_index'][1]]))
        drugs_kernels.append(t.DoubleTensor(getGipKernel(H3[:self.drug_size].clone(), 0, self.h3_gamma, True).double()))
        target_kernels.append(t.DoubleTensor(getGipKernel(H3[self.drug_size:].clone(), 0, self.h3_gamma, True).double()))

        H4 = t.relu(self.gat_4(H3, adj['edge_index'], adj['data'][adj['edge_index'][0], adj['edge_index'][1]]))
        drugs_kernels.append(t.DoubleTensor(getGipKernel(H4[:self.drug_size].clone(), 0, self.h4_gamma, True).double()))
        target_kernels.append(t.DoubleTensor(getGipKernel(H4[self.drug_size:].clone(), 0, self.h4_gamma, True).double()))

        drugs_kernels.append(self.drug_sim)
        target_kernels.append(self.target_sim)

        drug_k = sum([self.drug_ps[i] * drugs_kernels[i] for i in range(len(self.drug_ps))])
        self.drug_k = normalized_kernel(drug_k)
        target_k = sum([self.target_ps[i] * target_kernels[i] for i in range(len(self.target_ps))])
        self.target_k = normalized_kernel(target_k) 
        self.drug_l = laplacian(drug_k)
        self.target_l = laplacian(target_k)

        out1 = t.mm(self.drug_k, self.alpha1)
        out2 = t.mm(self.target_k, self.alpha2)

        out = (out1 + out2.T) / 2

        return out