import math
import pdb

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = math.sqrt(6.0 / (self.weight.shape[0] + self.weight.shape[1]))
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # pdb.set_trace()
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolutionCompress(Module):
    """
    GCN-compress layer
    """

    def __init__(self, in_features, out_features, mid, bias=False):
        super(GraphConvolutionCompress, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight1 = Parameter(torch.FloatTensor(in_features, mid))
        self.weight2 = Parameter(torch.FloatTensor(mid, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = math.sqrt(6.0 / (self.weight.shape[0] + self.weight.shape[1]))
        stdv1 = 1.0 / math.sqrt(self.weight1.size(1))
        stdv2 = 1.0 / math.sqrt(self.weight2.size(1))
        self.weight1.data.uniform_(-stdv1, stdv1)
        self.weight2.data.uniform_(-stdv2, stdv2)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv2, stdv2)

    def forward(self, input, adj):
        # pdb.set_trace()
        hidden = torch.mm(input, self.weight1)
        support = torch.mm(hidden, self.weight2)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
