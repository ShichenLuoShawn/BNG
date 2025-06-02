import torch, sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sn
import matplotlib.pyplot as plt
import random,math
from models import Baselines
from models.BNT.BNT.bnt import BrainNetworkTransformer

class KPFBNC(nn.Module):
    def __init__(self, num_of_node=200,num_of_pattern=400,topK=3,stride=25,classifier='ae',use_id='',device='cuda'):
        super().__init__()
        self.use_id = True if use_id else False
        self.device = device
        self.num_of_pattern = num_of_pattern
        self.num_of_node = num_of_node
        self.stride = stride
        self.topK = topK

        self.patterns = nn.Parameter(torch.Tensor(num_of_pattern, num_of_node))
        nn.init.xavier_uniform_(self.patterns.data, gain=1.414)

        self.maxpool = nn.MaxPool1d(self.stride,self.stride,return_indices=True)
        self.Graph_Analysis = Graph_Analysis(classifier,num_of_node,use_id=self.use_id,topK=topK)
        self.classifier = classifier

        ### reuse indice
        self.indice_p1 = torch.arange(num_of_pattern)[None,:,None]
        self.indice = torch.arange(360,device=self.device)[None,None,:].repeat(64,self.num_of_node,1)

        self.non_diag_scale = torch.ones(self.num_of_node,self.num_of_node,device=self.device)*topK*num_of_pattern\
        /num_of_node-torch.eye(self.num_of_node,device=self.device)*(topK*num_of_pattern/num_of_node-1)

    def forward(self, x, length, mode, epoch, args, label):
        size_after_pool = torch.floor((length[:,None,None])/self.stride).long()
        patterns = self.patterns

        normalized_patterns = patterns/torch.sqrt((patterns**2).sum(-1,keepdim=True)+1e-9)
        toppattern, patternmask = self.top_select(normalized_patterns,self.topK,-1)

        if args.data_name =='HCP_static':
            frames = x.transpose(2,1) # in HCP dataset all fmri have the same length, no padding zeros, no need to regularization
        else:
            frames = self.MRI_length_regularize(x,size_after_pool,length[:,None,None],self.stride,random_start=False)

        normalized_frames = frames/torch.sqrt((frames**2).sum(-1,keepdim=True)+1e-9)
        
        similarity_scores = torch.einsum('abc,dc->abd', normalized_frames, normalized_patterns)

        posscores, pos_indice = self.maxpool(similarity_scores.transpose(1,2))
        negscores, neg_indice = self.maxpool(-similarity_scores.transpose(1,2))
        compos = (posscores/(size_after_pool)).sum(-1)
        comneg = (negscores/(size_after_pool)).sum(-1)

        PES = (compos + comneg)/2
        
        key_patterns = toppattern.detach()
        graphs = (key_patterns.transpose(1,0))@torch.diag_embed(PES)@(key_patterns)*self.non_diag_scale

        norm = SigLoss(normalized_patterns,toppattern)
        orth = (Relaxed_orthloss(normalized_patterns,args.threshold))

        if self.classifier == 'ae':
            out, rec = self.Graph_Analysis(graphs,None,epoch)
        else:
            out, rec = self.Graph_Analysis(graphs,None,epoch), torch.tensor([0],device=self.device)

        return out, norm, orth, rec#torch.tensor([0],device=self.device)

    
    def MRI_length_regularize(self,frames,size_after_pool,length,stride,random_start):
        '''
            size_after_pool: (1d vector) size of scores after pooling, corresponds to the length of time series
            length: (1d vector) length of time series length of samples in a batch
            stride: (int) pooling stride
        '''
        mask = torch.zeros(frames.shape,device=self.device)
        if random_start:
            discarded = length-size_after_pool*stride-1
            discarded[discarded<=1] = 1
            start = random.choice([0,discarded])
            indice = self.indice[:frames.shape[0],:,:].clone()
            indice2 = indice+1
            indice[indice>=size_after_pool*stride+start] = 0
            indice[indice<start] = 0 
            indice2[indice2<start] = 0
            indice2[indice2>=start+330] = 0
            ind = indice2.nonzero()
            mask[torch.arange(mask.shape[0])[:,None,None],torch.arange(mask.shape[1])[None,:,None],indice] = 1
            newframe = (frames*mask)[ind[:,0],ind[:,1],ind[:,2]].reshape(mask.shape[0],mask.shape[1],-1)
        else:
            indice = self.indice[:frames.shape[0],:,:]+1
            indice[indice>=size_after_pool*stride] = 0
            mask[torch.arange(mask.shape[0])[:,None,None],torch.arange(mask.shape[1])[None,:,None],indice] = 1
            newframe = (frames*mask)[:,:,:(size_after_pool*stride).max()]
        return newframe.transpose(2,1)

    def top_select(self, inp, topk, dim, abs=True, return_mask=True):
        if abs:
            topinp, indice = torch.topk(torch.abs(inp),topk,dim)
        else:
            topinp, indice = torch.topk(inp,topk,dim)
        mask = torch.zeros(inp.shape,device=self.device)
        if len(inp.shape)==3:
            mask[torch.arange(inp.shape[0])[:,None,None],self.indice_p1,indice] = 1
        else:
            mask[torch.arange(inp.shape[0])[:,None],indice] = 1
        if return_mask:
            return mask*inp, mask
        else:
            return mask*inp


def Relaxed_orthloss(normalized_pattern, threshold, device='cuda'):
    sim = (normalized_pattern@normalized_pattern.T)*(1-torch.eye(normalized_pattern.shape[0],device=device))
    return (F.relu((1/(1-threshold+1e-5))*(torch.abs(sim)-threshold))**2).mean()

def SigLoss(p, topp): #make focused regions more significant in patterns
    lenp = torch.norm(p,2,dim=-1) #*(1-torch.eye(p.shape[0],device='cuda'))
    lentopp = torch.norm(topp,2,dim=-1) 
    return ((1-(lentopp/lenp))**2).mean()

class Graph_Analysis(nn.Module):
    def __init__(self, model_name='mlp',num_of_node=200,hidden=64,cout=2,use_id=False,topK=3):
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
        elif model_name == 'BNT':
            self.model = BrainNetworkTransformer(self.in_c,self.in_c//topK)

    def forward(self,x,identity,epoch):
        if  self.model_name=='ae':
            out, rec = self.model(x)
            return out, rec
        elif self.model_name=='mlp' or self.model_name=='BNT':
            out = self.model(x)
            return out
        elif self.model_name in ['gat','gcn','gin']:
            inp = torch.cat([x,identity[None,:,:].repeat(x.shape[0],1,1)],-1) if self.use_id else x
            out = self.model(inp,x)
            return out

def weight_init(m):
    for i in m.modules():
        if isinstance(i,nn.Linear) or isinstance(i, nn.Conv1d):
            nn.init.xavier_normal_(i.weight.data)
            if i.bias is not None:
                nn.init.uniform_(i.bias,-0.01,0.01)

def show_focused_signal(frames,posindice,negindice,toppaterns,mask,roiindice):
    p_indice,n_indice,toppaterns = posindice.clone().detach().cpu()[0], negindice.clone().detach().cpu()[0],toppaterns.clone().detach().cpu()
    mask = mask.detach().cpu()
    frames = frames[0].T.detach().cpu()
    selected_roi_index = roiindice
    selected_roi = torch.zeros(200)
    selected_roi[selected_roi_index] = 1
    # toppaterns[toppaterns<=0.35] = 0
    pattern_index = (toppaterns*selected_roi).sum(-1).nonzero().flatten()
    roi_2_ind = {int(selected_roi_index[i]):i for i in range(len(selected_roi_index))}
    print('selected ROI:',roi_2_ind,'involved num of pattern:',len(pattern_index))

    colors = ['olive', 'green', 'blue','orange','c','olive', 'green', 'blue','orange','c','red']
    length = 100
    x = [i for i in range(100)]
    meanframes = frames[selected_roi_index].mean(-1).detach().cpu().numpy()
    frames = frames[selected_roi_index,0:100].detach().cpu().numpy()
    all_value = np.abs(frames).mean()
    first = 0
    plt.figure(figsize=(8, 8), dpi=200,facecolor='lightgray')
    for i,roi in enumerate(selected_roi_index):
        if first == 0:
            first+=1
            # plt.scatter(x,frames[i]-i*5,s=5,color='red',label='fMRI signals')
            plt.plot(x,frames[i]-i*4.5,color=colors[i],label='fMRI signal sequences')
            plt.axhline(y=(meanframes[i]-4.5*i).mean(), color=colors[i], linestyle='--',linewidth=1)
        else:
            plt.plot(x,frames[i]-i*4.5,color=colors[i])
            # plt.scatter(x,frames[i]-i*5,s=5,color='red')
            plt.axhline(y=(meanframes[i]-4.5*i).mean(), color=colors[i], linestyle='--',linewidth=1)
    first = 0
    counter = 0
    total_value = 0
    for i,ind in enumerate(pattern_index):
        x_co = p_indice[ind,0:4]
        x_co2 = n_indice[ind,0:4]
        all_roi = mask[ind].nonzero().flatten()
        involved_roi = np.intersect1d(all_roi, selected_roi_index)
        for roi in involved_roi:

            y = frames[roi_2_ind[roi],x_co]-4.5*roi_2_ind[roi]
            y2 = frames[roi_2_ind[roi],x_co2]-4.5*roi_2_ind[roi]

            counter = counter + x_co.shape[0] + x_co2.shape[0]
            total_value = total_value + np.abs(y+4.5*roi_2_ind[roi]).sum() + np.abs(y2+4.5*roi_2_ind[roi]).sum()

            if first == 0:
                first+=1
                plt.scatter(x_co,y,color=colors[-1],s=24,label='Focused Signals')
            else:
                plt.scatter(x_co,y,color=colors[-1],s=24)
                plt.scatter(x_co2,y2,color=colors[-1],s=24)

    print(total_value/counter)
    print(all_value)

    plt.tick_params(axis='both', which='both', length=0)
    plt.ylim(-45,6)
    plt.legend(loc='upper left',frameon=True, fancybox=True, shadow=True)
    plt.tight_layout
    plt.show()
