import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio
import torch.nn.functional as F
import pdb
l1_lambda=0.01
#########MAHGCN
class MAHGCN(nn.Module):
    def __init__(self, act, drop_p, layer):
        super(MAHGCN, self).__init__()
        self.layer = layer
        self.net_gcn_down1 = GCN(392, 1, act, drop_p)
        self.net_pool1 = AtlasMap(392, 200, drop_p)
        self.net_gcn_bot = GCN(200, 1, act, drop_p)

    def forward(self, g1, g2, h):
        h = self.net_gcn_down1(g2, h)
        downout1 = h
        h = self.net_pool1(h)
        h = self.net_gcn_bot(g1, torch.diag(h))
        hh = torch.cat((h, downout1))
        return hh
        
    #def l1_regularization(self):
        #l1_reg = torch.tensor(0.)
        #for param in self.parameters():
            #l1_reg += torch.sum(torch.abs(param))
        #return self.l1_lambda * l1_reg
    
    def custom_loss(self):
        l1_regularization = 0.1  # Adjust the regularization strength
        for param in self.parameters():
            l1_loss = l1_regularization * sum(abs(coef) for coef in lasso.coef_)
            #mse_loss = mean_squared_error(y_true, y_pred)
        return self.l1_loss

class AtlasMap(nn.Module):

    def __init__(self, indim, outdim, p):
        super(AtlasMap, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h):
        h = h.T
        filename = 'interlayermapping/mapping_Li.mat'
        Map = scio.loadmat(filename)
        Map = Map['ratio_mapping']
        Map = torch.tensor(Map)
        Map = Map.float()
        Map = Map.cuda()
        h = torch.matmul(h, Map)
        h = h.T
        h = torch.squeeze(h)
        return h


class AtlasMap_mean(nn.Module):

    def __init__(self, indim, outdim, p):
        super(AtlasMap_mean, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h):
        h = h.T
        filename = 'interlayermapping/mapping_Li.mat'
        Map = scio.loadmat(filename)
        Map = Map['mapping']
        Map = torch.tensor(Map)
        Map = Map.float()
        Map = Map.cuda()
        Map = Map / torch.sum(Map, axis=0)
        h = torch.matmul(h, Map)
        h = h.T
        h = torch.squeeze(h)
        return h

class AtlasMap_max(nn.Module):

    def __init__(self, indim, outdim, p):
        super(AtlasMap_max, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h):
        h = h.T
        dim=h.shape
        filename = 'interlayermapping/mapping_Li.mat'
        Map = scio.loadmat(filename)
        Map = Map['mapping']
        Map = torch.tensor(Map)
        Map = Map.float()
        Map = Map.cuda()

        h = h.T * Map
        h = torch.max(h, axis=0).values
        h = torch.reshape(h,(dim[0],self.outdim))
        h = h.T
        h = torch.squeeze(h)
        return h

class AtlasMap_th(nn.Module):

    def __init__(self, indim, outdim, p, th):
        super(AtlasMap_th, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.th=th

    def forward(self, h):
        h = h.T
        filename = 'interlayermapping/mapping_Li.mat'
        Map = scio.loadmat(filename)
        Map = Map['mapping']
        Map[Map<self.th] =0
        Map[Map>= self.th] = 1
        Map = torch.tensor(Map)
        Map = Map.float()
        Map = Map.cuda()
        h = torch.matmul(h, Map)
        h = h.T
        h = torch.squeeze(h)
        #h = torch.diag(h)
        return h


class GraphUnet(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.top_gcn = GCN(in_dim, out_dim, act, drop_p)
        self.down_gcns = nn.ModuleList()
        #self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        #self.unpools = nn.ModuleList()
        self.dim=dim
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.pools.append(Pool(ks[i], dim, drop_p))
        self.down_gcns.append(GCN(392, 1, act, drop_p))
        self.down_gcns.append(GCN(200, 1, act, drop_p))

    def forward(self, g, h):
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        h = self.top_gcn(g, h)
        h1 = h
        for i in range(self.l_n):
            g, h, idx = self.pools[i](g, h)
            h = self.down_gcns[i](g, torch.diag(torch.squeeze(h)))
            h1=torch.cat([h1, h], dim=0)
        return h1


class GraphDiif(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        super(GraphDiif, self).__init__()
        self.ks = ks
        self.top_gcn = GCN(in_dim, out_dim, act, drop_p)
        self.down_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.dim=dim
        self.l_n = len(ks)
        self.pools.append(DiffPool(329, 200))
        self.down_gcns.append(GCN(200, 1, act, drop_p))

    def forward(self, g, h):
        hnext = self.top_gcn(g, h)
        h1 = hnext
        for i in range(self.l_n):
            g, h = self.pools[i](g, h, hnext)
            h=torch.diag(torch.squeeze(h))
            hnext = self.down_gcns[i](g, h)
            h1=torch.cat([h1, hnext], dim=0)

        return h1
    def Loss(self):
        L=0
        for i in range(self.l_n):
            L=L+self.pools[i].Loss()
        return L

class DiffPool(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DiffPool, self).__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.gcn=GCN(self.in_dim, self.out_dim, nn.ReLU())
        self.softmax = nn.Softmax()
        self.assign = torch.zeros(in_dim, out_dim)
        self.g = torch.zeros(in_dim, in_dim)

    def forward(self, g, h, hnext):
        self.g = g
        self.assign=self.gcn(g,h)
        #print(assign.shape)
        self.assign=self.softmax(self.assign)
        newh = torch.matmul(torch.transpose(self.assign, 0, 1), hnext)
        newg = torch.transpose(self.assign, 0, 1) @ g @ self.assign
        return newg, newh
    def Loss(self):
        loss_LP=torch.norm(torch.cat([self.g,torch.matmul(self.assign,torch.transpose(self.assign, 0, 1))]))/self.in_dim
        loss_en=0
        eps = 1e-7
        NPassign=F.relu(self.assign)
        for row in range(self.in_dim):
            loss_en=loss_en-torch.sum(NPassign[row,:]*torch.log(NPassign[row,:])+eps)
        loss_en=loss_en/self.in_dim
        return (loss_LP+loss_en)/self.in_dim

class GraphSAG(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        super(GraphSAG, self).__init__()
        self.ks = ks
        self.top_gcn = GCN(in_dim, out_dim, act, drop_p)
        self.down_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.dim=dim
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.pools.append(Pool(ks[i], dim, drop_p))
        self.down_gcns.append(GCN(200, 1, act, drop_p))

    def forward(self, g, h):
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        h = self.top_gcn(g, h)
        h1 = h
        for i in range(self.l_n):
            g, h, idx = self.pools[i](g, h)
            h = self.down_gcns[i](g, torch.diag(torch.squeeze(h)))
            h1=torch.cat([h1, h], dim=0)
        return h1

class SAGPool(nn.Module):
    def __init__(self, k, in_dim, p):
        super(SAGPool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = GCN(self.in_dim, 1, nn.ReLU())
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)
#############GCN
import torch
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, act, p=0.3):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
        self.l1_lambda = l1_lambda  # Regularization strength

    def forward(self, g, h):
        h = self.drop(h)
        h = torch.matmul(g, h)
        h = self.proj(h)
        h = self.act(h)
        return h

    #def l1_regularization(self):
        #l1_reg = torch.tensor(0.)
        #for param in self.parameters():
         #   l1_reg += torch.sum(torch.abs(param))
        #return self.l1_lambda * l1_reg

    def custom_loss(self):
        l1_regularization = 0.1  # Adjust the regularization strength
        for param in self.parameters():
            l1_loss = l1_regularization * sum(abs(coef) for coef in lasso.coef_)
            #mse_loss = mean_squared_error(y_true, y_pred)
        return self.l1_loss

#############GCN

class Pool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h


def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx


def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g



    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)
