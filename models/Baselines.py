import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_geometric
from torch_geometric.nn import Sequential, GCNConv, GINConv


class mlp(nn.Module):
    def __init__(self,inchannel):
        super().__init__()
        self.twolayerMLP = nn.Sequential(nn.Linear(inchannel**2,200),nn.ReLU(),(nn.Linear(200,2)))
    
    def forward(self, x):
        shape = x.shape
        out = self.twolayerMLP(x.reshape(shape[0],-1))
        return out

class AE(nn.Module):
    def __init__(self,inchannel):
        super().__init__()
        self.enc1 = nn.Linear(inchannel**2//2+inchannel//2,512)
        self.enc2 = nn.Linear(512,128)
        self.dec2 = nn.Linear(128,512)
        self.dec1 = nn.Linear(512,inchannel**2//2+inchannel//2)
        self.act = nn.ReLU()
        self.fcout = nn.Linear(128,2)

    def forward(self,x):
        '''graph2vector'''
        shape = x.shape
        x[x==0] = 1e-9

        x = x.tril().reshape(shape[0],-1)
        x = x[x!=0].reshape(shape[0],-1)
        out1 = self.act(self.enc1(x))
        out2 = self.enc2(out1)
        rec_x = self.dec1(self.act(self.dec2(out2)))
        out = self.fcout(self.act(out2))
        rec =0.1*((rec_x-x)**2).mean()
        return out,rec

    

class GCN(nn.Module):
    def __init__(self,c_in=200,hidden=128,c_out=2,num_of_node=200):
        super().__init__()
        self.gcnlayer1 = GCNConv(c_in,hidden)
        self.gcnlayer2 = GCNConv(hidden,hidden)
        self.act = nn.ReLU()
        self.num_of_node = num_of_node
        self.out = nn.Linear(hidden,c_out)
    
    def forward(self,x,adj,a=None,b=None,c=None,d=None):
        value,indice = torch.topk(torch.abs(x.reshape(x.shape[0],40000)),int(40000*0.3),-1)
        mask = torch.zeros(x.shape[0],200*200,device='cuda')
        mask[torch.arange(x.shape[0])[:,None],indice] = 1
        adj = x*mask.reshape(x.shape[0],200,200)
        shape = x.shape
        edge = adj.nonzero()
        batch = edge[:,0]*self.num_of_node
        edge_index = (edge[:,[1,2]]+batch[:,None]).T
        x = x.reshape(shape[0]*shape[1],-1)
        out = self.act(self.gcnlayer1(x,edge_index))
        out = self.act(self.gcnlayer2(out,edge_index))
        out = out.reshape(shape[0],shape[1],128)
        out = out.mean(-2)
        out = self.out(out)
        return out#,torch.tensor([0],device='cuda'),torch.tensor([0],device='cuda')

class GIN(nn.Module):
    def __init__(self,c_in=200,hidden=128,c_out=2,num_of_node=200):
        super().__init__()
        self.ginlayer1 = GINConv(nn.Linear(c_in,hidden),train_eps=True)
        self.ginlayer2 = GINConv(nn.Linear(hidden,c_out),train_eps=True)
        self.act = nn.ReLU()
        self.num_of_node = num_of_node

    def forward(self,x,adj):
        shape = x.shape
        edge = adj.nonzero()
        batch = edge[:,0]*self.num_of_node
        edge_index = (edge[:,[1,2]]+batch[:,None]).T
        x = x.reshape(shape[0]*shape[1],-1)
        out = self.act(self.ginlayer1(x,edge_index))
        out = self.ginlayer2(out,edge_index)
        out = out.reshape(shape[0],shape[1],2)
        out = out.mean(-2)
        return out