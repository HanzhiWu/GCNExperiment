import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphConvolutionCompress


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, sampler, compress):
        super().__init__()
        if compress == 1.0:
            self.gc1 = GraphConvolution(nfeat, nhid)
        else:
            n1 = int(compress * nfeat * nhid / (nfeat + nhid))
            self.gc1 = GraphConvolutionCompress(nfeat, nhid, n1)

        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.sampler = sampler
        # self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x, adj):
        outputs1 = F.relu(self.gc1(x, adj[0]))
        outputs1 = F.dropout(outputs1, self.dropout, training=self.training)
        outputs2 = self.gc2(outputs1, adj[1])
        return F.log_softmax(outputs2, dim=1)
        # return self.out_softmax(outputs2)

    def sampling(self, *args, **kwargs):
        return self.sampler.sampling(*args, **kwargs)


class GCN_normal(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, compress):
        super().__init__()
        if compress == 1.0:
            self.gc1 = GraphConvolution(nfeat, nhid)
        else:
            n1 = int(compress * nfeat * nhid / (nfeat + nhid))
            self.gc1 = GraphConvolutionCompress(nfeat, nhid, n1)

        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        # self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x, adj):
        outputs1 = F.relu(self.gc1(x, adj))
        outputs1 = F.dropout(outputs1, self.dropout, training=self.training)
        outputs2 = self.gc2(outputs1, adj)
        return F.log_softmax(outputs2, dim=1)
