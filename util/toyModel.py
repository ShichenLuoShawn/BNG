import torch, sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sn
import matplotlib.pyplot as plt
import random,math
from models import Baselines
from models import graphMask
from models.Components import *

class KPbyNode(nn.Module):
    def __init__(self, num_of_node=200,num_of_pattern=200,topK=3,classifier='ae',use_id=''):
        super().__init__()
        self.use_id = True if use_id[0:2] == 'id' else False
        self.device='cuda'
        self.num_of_pattern = num_of_pattern
        self.num_of_node = num_of_node
        self.stride = 25
        self.topK = topK
        self.act = nn.ReLU()

        self.patterns = nn.Parameter(torch.Tensor(num_of_pattern, num_of_node))
        nn.init.xavier_uniform_(self.patterns.data, gain=1.414)

        self.maxpool = nn.MaxPool1d(self.stride,self.stride,return_indices=True)
        self.Graph_Analysis = Graph_Analysis(classifier,num_of_node,use_id=self.use_id)

        ### reuse indice
        self.p0 = torch.arange(num_of_pattern)[:,None]
        self.p1 = torch.arange(num_of_pattern)[None,:,None]
        self.indice = torch.arange(360,device=self.device)[None,None,:].repeat(64,self.num_of_node,1)

        self.non_diag_scale = torch.ones(self.num_of_node,self.num_of_node,device=self.device)*self.topK-torch.eye(self.num_of_node,device=self.device)*(self.topK-1)

        self.reconstruct_model = graphMask.graphMask(num_of_node,128,topK)
        self.flag = 0

        self.out = nn.Linear(128,2)

    def forward(self, x, length, mode, epoch, args):
        size_after_pool = torch.floor((length[:,None,None])/self.stride).long()
        # patterns = self.patterns
        patterns = F.tanh((self.patterns)*(1-torch.eye(self.num_of_node,device=self.device)))+torch.eye(self.num_of_node,device=self.device)
        normalized_patterns = patterns/torch.sqrt((patterns**2).sum(-1,keepdim=True)+1e-9)

        frames = self.MRI_length_regularize(x,size_after_pool,length[:,None,None],self.stride,random_start=True)
        # frames = x.transpose(2,1)
        normalized_frames = frames/torch.sqrt((frames**2).sum(-1,keepdim=True)+1e-9)
        
        scores = torch.einsum('abc,dc->abd', normalized_frames, normalized_patterns)

        posscores, pos_indice = self.maxpool(scores.transpose(1,2))
        negscores, neg_indice = self.maxpool(-scores.transpose(1,2)) 
        p_indice,n_indice = pos_indice.clone().detach(), neg_indice.clone().detach()
        # if mode == 'train' and random.random()<0.5:
        #     posscores,negscores,size_after_pool,signal_mask = self.random_signal_mask(posscores,negscores,size_after_pool)
        #     signal_mask = signal_mask.repeat(1,200,1)
        #     p_indice[signal_mask==0] = frames.shape[1]-1
        #     n_indice[signal_mask==0] = frames.shape[1]-1

        signal_select = torch.zeros(frames.shape,device=self.device).transpose(2,1)
        signal_select[torch.arange(frames.shape[0])[:,None,None], self.p1, p_indice] += 1
        signal_select[torch.arange(frames.shape[0])[:,None,None], self.p1, n_indice] += -1

        # pos_indice = pos_indice+(torch.arange(x.shape[0],device='cuda')[:,None,None]*normalized_frames.shape[-2])
        # pos_selected_signals = torch.index_select(normalized_frames.reshape(-1,200),0,pos_indice.flatten())
        # pos_selected_signals = pos_selected_signals.reshape(x.shape[0],-1,200,200)*math.sqrt(200/self.topK)

        # neg_indice = neg_indice+(torch.arange(x.shape[0],device='cuda')[:,None,None]*normalized_frames.shape[-2])
        # neg_selected_signals = torch.index_select(normalized_frames.reshape(-1,200),0,neg_indice.flatten())
        # neg_selected_signals = neg_selected_signals.reshape(x.shape[0],-1,200,200)*math.sqrt(200/self.topK)

        avg_selected_signal = math.sqrt(200/self.topK)*signal_select@normalized_frames/size_after_pool/2

        # avg_signal = torch.abs(frames).sum(-2,keepdim=True)/(size_after_pool*self.stride)
        
        compos = (posscores/(size_after_pool)).sum(-1)#-scores.mean(-2)
        comneg = (negscores/(size_after_pool)).sum(-1)#+scores.mean(-2)

        finalscore = (compos + comneg)/2#-torch.sqrt(size_after_pool2[:,None])*0.001

        toppattern, patternmask = self.top_select(normalized_patterns,self.topK,-1)
        Tframe = avg_selected_signal*patternmask

        scores2 = (Tframe/torch.sqrt((Tframe**2).sum(-1,keepdim=True)+1e-9) * toppattern/torch.sqrt((toppattern**2).sum(-1,keepdim=True)+1e-9)).sum(-1)
        # if epoch%20==0:
        #     print(scores2.max(),scores2.min(),scores2.mean(),scores2.shape,((1-scores2)**2).mean())
        sss = ((1-F.relu(scores2))**2).mean()
        # pos_selected_signals = pos_selected_signals*patternmask
        # neg_selected_signals = neg_selected_signals*patternmask


        norm = SigLoss(normalized_patterns,toppattern)
        identity = (toppattern@toppattern.transpose(1,0)).detach()
        
        # realpattern = valuesqrt(toppattern).detach()
        realpattern = toppattern.clone().detach()
        realpattern[realpattern>0] = 1/self.topK
        realpattern[realpattern<0] = -1/self.topK
        # realpatternpos = valuesqrt(pos_selected_signals).detach()
        # realpatternneg = valuesqrt(neg_selected_signals).detach()
        
        graphs = (realpattern.transpose(1,0))@torch.diag_embed(finalscore+0.25*(scores2*math.sqrt(self.topK/200)))@(realpattern)*self.non_diag_scale/(self.topK/5)
        # graphs = (realpatternpos.transpose(3,2))@torch.diag_embed(posscores.transpose(2,1))@(realpatternpos)*self.non_diag_scale/(self.topK/5)
        # graphs += (realpatternneg.transpose(3,2))@torch.diag_embed(negscores.transpose(2,1))@(realpatternneg)*self.non_diag_scale/(self.topK/5)
        # graphs = graphs.sum(1)/size_after_pool/2

        # mode = 'traint' if args.spec!='rec' else mode   
        # out, rec_graph, masked_graphs = self.reconstruct_model(toppattern,finalscore,avg_selected_signal,self.non_diag_scale,3,mode)
        # recloss = 0.1*((rec_graph-(masked_graphs-graphs))**2).mean()

        # if epoch==41 and self.flag==0:
        #     print(finalscore.mean(),scores2.mean())
        #     self.flag=1 
        #     figs,ax = plt.subplots(2,2)
        #     sn.heatmap((toppattern).detach().cpu().numpy(),center=0,ax=ax[0,0])
        #     sn.heatmap(graphs[0].detach().cpu().numpy(),center=0,ax=ax[0,1])
        #     sn.heatmap(graphs[1].detach().cpu().numpy(),center=0,ax=ax[1,0])
        #     sn.heatmap(graphs[2].detach().cpu().numpy(),center=0,ax=ax[1,1])
        #     plt.show()

        out = self.Graph_Analysis(graphs,identity)
        # out = self.out(out)
        return out, norm, graphs

    
    def MRI_length_regularize(self,frames,size_after_pool,length,stride,random_start):
        '''
            size_after_pool: (1d vector) size of scores after pooling, corresponds to the length of time series
            length: (1d vector) length of time series length of samples in a batch
            stride: (int) pooling stride
        '''
        newframe=frames[:,:,:size_after_pool*stride]
        return newframe.transpose(2,1)

    def top_select(self, inp, topk, dim, abs=True, return_mask=True):
        if abs:
            topinp, indice = torch.topk(torch.abs(inp),topk,dim)
        else:
            topinp, indice = torch.topk(inp,topk,dim)
        mask = torch.zeros(inp.shape,device=self.device)
        if len(inp.shape)==3:
            mask[torch.arange(inp.shape[0])[:,None,None],self.p1,indice] = 1
        else:
            mask[self.p0,indice] = 1
        if return_mask:
            return mask*inp, mask
        else:
            return mask*inp
    
    def random_signal_mask(self, posscores, negscores, size_after_pool):
        mask = torch.arange(posscores.shape[2],device=self.device)[None,None,:].repeat(posscores.shape[0],1,1)
        mask[mask<size_after_pool] = 1
        mask[mask>=size_after_pool] = 0
        randmask = torch.rand(posscores.shape[0],1,posscores.shape[2],device=self.device)
        randrange = 0.2+random.random()/3
        randmask[randmask<=randrange] = 0
        randmask[randmask!=0] = 1
        finalmask = mask*randmask
        finalmask[finalmask.sum(-1)<3] = mask[finalmask.sum(-1)<3].float()
        new_size_after_pool = finalmask.sum(-1,keepdim=True)
        return finalmask*posscores, finalmask*negscores, new_size_after_pool, finalmask


def show_heatmap(tensor,center=0):
    tensor = tensor.clone().detach().cpu().numpy()
    sn.heatmap(tensor)
    plt.show()

def valuesqrt(x):
    sign = torch.ones(x.shape,device='cuda')
    sign[x<=0] = -1
    sign[x==0] = 0
    newx = torch.sqrt(torch.abs(x)+1e-10)*sign
    return newx

def Relorthloss(m, threshold, device='cuda'):
    pattern_mods = torch.sqrt((m**2).sum(-1,keepdim=True)+1e-9)
    mods_prod = pattern_mods@pattern_mods.T+1e-9
    threshold = torch.ones(m.shape[0],m.shape[1],device='cuda')*threshold
    maskorth = torch.ones(m.shape[0],m.shape[1],device='cuda')
    maskorth[m@m.T/mods_prod<=threshold] = 0
    return torch.norm((m@m.T)/mods_prod*maskorth*\
                (1-torch.eye(m.shape[0],m.shape[1],device='cuda')),p='fro')

def SigLoss(p, topp): #make focused regions more significant in patterns
    lenp = torch.norm(p,2)
    lentopp = torch.norm(topp,2)
    return ((1-lentopp/lenp)**2).mean()

class Graph_Analysis(nn.Module):
    def __init__(self, model_name='mlp',num_of_node=200,hidden=64,cout=2,use_id=False):
        super().__init__()
        self.model_name = model_name
        self.num_of_node = num_of_node
        self.in_c = 2*num_of_node if use_id else num_of_node
        self.use_id = use_id
        if model_name=='mlp':
            self.model = Baselines.mlp(self.in_c)
        elif model_name=='gat':
            self.model = Baselines.GAT(self.in_c,hidden,cout)
        elif model_name=='ae':
            self.model = Baselines.AE(self.in_c)
        elif model_name=='gcn':
            self.model = Baselines.GCN(self.in_c)
        elif model_name=='gin':
            self.model = Baselines.GIN(self.in_c)
        elif model_name == 'hgcn':
            self.model = Baselines.HGCNModel()

    def forward(self,x,identity):
        if self.model_name=='mlp' or self.model_name=='ae':
            out = self.model(x)
        elif self.model_name in ['gat','gcn','gin','hgcn']:
            inp = torch.cat([x,identity[None,:,:].repeat(x.shape[0],1,1)],-1) if self.use_id else x
            out = self.model(inp,x)
        return out


def weight_init(m):
    for i in m.modules():
        if isinstance(i,nn.Linear) or isinstance(i, nn.Conv1d):
            nn.init.xavier_normal_(i.weight.data)
            if i.bias is not None:
                nn.init.uniform_(i.bias,-0.01,0.01)  