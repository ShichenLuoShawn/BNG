from torch.utils.data import Dataset
from torch.utils.data import dataloader
import torch
import torch.nn.functional as F
from scipy.io import loadmat,savemat
import numpy as np
import pandas as pd
import os, nilearn, sys
import matplotlib.pyplot as plt
import seaborn as sn
import random, re
from scipy.stats import kstest, norm

class MyDatasetRaw(Dataset):
    def __init__(self,path,option='ts',device='cpu'):
        self.device=device
        self.op = option
        self.timeSeries = []
        self.labels = []
        self.length = []
        self.site=[]
        self.path = path
        self.metadata = pd.read_csv(f'{path}\\'+'label.csv')[["subject","DX_GROUP","SITE_ID"]] if path[-6:] != 'static' else pd.read_csv(f'{path}\\'+'label.csv')[['Subject','Gender']]
        self.sites = pd.unique((self.metadata['SITE_ID'])) if path[-6:] != 'static' else [1]
        self.sites = {i:[] for i in self.sites}
        self.LoadData(pearson=False)

    def __len__(self):
        return len(self.timeSeries)
    
    def __getitem__(self, index):
        return self.timeSeries[index],self.labels[index],self.length[index]
    
    def LoadData(self, pearson=False, top25=True):
        for root, dirs, files in os.walk(self.path):
            count=0
            for file in files:
                if file.endswith('cc200.1D'):
                    timeSeries = torch.Tensor(np.loadtxt(root+'\\'+file,dtype=float).transpose(1,0)).to(self.device)
                    length = timeSeries.shape[1]
                    paddingsize = 360-length

                    timeSeries = (timeSeries-timeSeries.mean(0,keepdim=True))/(timeSeries.std(0,keepdim=True)+1e-9)
                    timeSeries = (timeSeries-timeSeries.mean(-1,keepdim=True))/(timeSeries.std(-1,keepdim=True)+1e-9)

                    timeSeries = F.pad(timeSeries,(0,paddingsize),'constant',0)[None,:,:]
                    ID = file[-19:-14]
                    label = self.metadata.loc[self.metadata['subject']==int(ID)].iloc[0,1]-1 # 1 autism 0 control
                    site = self.metadata.loc[self.metadata['subject']==int(ID)].iloc[0,2]
                    self.sites[site].append(count)
                elif file.endswith('cc200.csv'):
                    ID = int(re.search('^[0-9]*_',file).group()[0:-1])
                    label = self.metadata.loc[self.metadata['subject']==int(ID)].iloc[0,1]
                    if label == 'withheld': continue
                    else: label = int(label)
                    label = 1 if label>=1 else 0
                    timeSeries = torch.Tensor(np.array(pd.read_csv(root+'\\'+file,dtype=float)).transpose(1,0)).to(self.device)[1:,:]
                    length = timeSeries.shape[1]
                    paddingsize = 360-length
                    timeSeries = (timeSeries-timeSeries.mean(-1,keepdim=True))/(timeSeries.std(-1,keepdim=True)+1e-9)
                    timeSeries = F.pad(timeSeries,(0,paddingsize),'constant',0)[None,:,:]
                    site = self.metadata.loc[self.metadata['subject']==int(ID)].iloc[0,2]
                    self.sites[site].append(count)
                elif file.endswith('.txt'):
                    ID = file[:6]
                    label = self.metadata.loc[self.metadata['Subject']==int(ID)].iloc[0,1]
                    label = 1 if label=='M' else 0
                    timeSeries = torch.Tensor(np.loadtxt(root+'\\'+file,dtype=float)).to(self.device).transpose(1,0)
                    length = timeSeries.shape[-1]
                    timeSeries = ((timeSeries-timeSeries.mean(-1,keepdim=True))/(timeSeries.std(-1,keepdim=True)+1e-9))[None,:,:]
                    site=1
                    self.sites[site].append(count)
                else:
                    continue
                
                if self.op=='pc':
                    corr = timeSeries@timeSeries.transpose(2,1)/length
                    self.timeSeries.append(corr/corr[:,0,0])
                else:
                    self.timeSeries.append(timeSeries)
                self.labels.append(label)
                self.length.append(length)
                count+=1

        self.labels=torch.Tensor(self.labels).to(self.device)
        self.timeSeries=torch.cat(self.timeSeries).to(self.device)
        self.length = torch.tensor(self.length).to(self.device)
    
class DataSpliter():
    def __init__(self,data,NumofFold:int,testRate=1,RSeed = 0):
        assert (testRate<=1 and testRate>0), 'testRate must within (0,1]'
        assert type(NumofFold)==int, 'NumofFold must be an integer'
        self.Rseed = RSeed
        self.data = data
        self.testRate = testRate
        self.NumofFold = NumofFold
        # self.pos_index,self.neg_index,self.testsize_pos,self.testsize_neg = self.GetIndex_stratifiled()
        self.index,self.testsize = self.GetIndex()

    # def GetTest(self, Batchsize, FoldIndex, shuffle=False):
    #     testindex = np.concatenate([self.pos_index[FoldIndex*self.testsize_pos:(FoldIndex+1)*self.testsize_pos],self.neg_index[FoldIndex*self.testsize_neg:(FoldIndex+1)*self.testsize_neg]])
    #     testset = torch.utils.data.Subset(self.data, testindex)
    #     return torch.utils.data.DataLoader(testset, batch_size=Batchsize, shuffle=shuffle)
    def GetTest(self, Batchsize, FoldIndex, shuffle=False):
        testindex = self.index[FoldIndex*self.testsize:(FoldIndex+1)*self.testsize]
        testset = torch.utils.data.Subset(self.data, testindex)
        return torch.utils.data.DataLoader(testset, batch_size=Batchsize, shuffle=shuffle)

    def GetTrain(self, Batchsize, FoldIndex, shuffle=True, eval=True):
        if not eval:
            trainindex = np.concatenate([self.pos_index[:FoldIndex*self.testsize_pos],self.pos_index[(FoldIndex+1)*self.testsize_pos:],\
                                         self.neg_index[:FoldIndex*self.testsize_neg],self.neg_index[(FoldIndex+1)*self.testsize_neg:]])
            trainset = torch.utils.data.Subset(self.data, trainindex)
            return torch.utils.data.DataLoader(trainset, batch_size=Batchsize, shuffle=shuffle), 0
        else:
            trainindex = np.concatenate([self.index[:FoldIndex*self.testsize],self.index[(FoldIndex+1)*self.testsize:]])
            # evalindex = trainindex[:self.testsize//2]
            # trainindex = trainindex[self.testsize//2:]
            trainset = torch.utils.data.Subset(self.data, trainindex)
            # evalset = torch.utils.data.Subset(self.data, evalindex)
            return torch.utils.data.DataLoader(trainset, batch_size=Batchsize, shuffle=shuffle), 0#torch.utils.data.DataLoader(evalset, batch_size=Batchsize, shuffle=shuffle)
        
    # def GetTrain(self, Batchsize, FoldIndex, shuffle=True, eval=True):
    #     if not eval:
    #         trainindex = np.concatenate([self.index[:FoldIndex*self.testsize],self.index[(FoldIndex+1)*self.testsize:]])
    #         trainset = torch.utils.data.Subset(self.data, trainindex)
    #         return torch.utils.data.DataLoader(trainset, batch_size=Batchsize, shuffle=shuffle), 0
    #     else:
    #         trainindex = np.concatenate([self.index[:FoldIndex*self.testsize],self.index[(FoldIndex+1)*self.testsize:]])
    #         evalindex = trainindex[:self.testsize//2]
    #         trainindex = trainindex[self.testsize//2:]
    #         trainset = torch.utils.data.Subset(self.data, trainindex)
    #         evalset = torch.utils.data.Subset(self.data, evalindex)
    #         return torch.utils.data.DataLoader(trainset, batch_size=Batchsize, shuffle=shuffle), torch.utils.data.DataLoader(evalset, batch_size=Batchsize, shuffle=shuffle)

    def GetIndex(self):
        DataLength = len(self.data)
        index = np.random.permutation(np.array([i for i in range(DataLength)]))
        if self.testRate==1:
            testsize = DataLength//self.NumofFold
        else:
            testsize = int(DataLength//self.NumofFold*self.testRate)
        return index, testsize
    
    def GetIndex_LABEL_stratifiled(self):
        pos_index = self.data.labels.nonzero().flatten().cpu().numpy()
        neg_index = (1-self.data.labels).nonzero().flatten().cpu().numpy()
        pos_index = np.random.permutation(pos_index)
        neg_index = np.random.permutation(neg_index)
        testsize_pos = len(pos_index)//5
        testsize_neg = len(neg_index)//5
        return pos_index, neg_index, testsize_pos, testsize_neg

class DataSpliter_Stratified():
    def __init__(self,data,NumofFold:int,testRate=1,RSeed = 0):
        assert (testRate<=1 and testRate>0), 'testRate must within (0,1]'
        assert type(NumofFold)==int, 'NumofFold must be an integer'
        self.Rseed = RSeed
        self.data = data
        self.testRate = testRate
        self.NumofFold = NumofFold
        self.index,self.testsize = self.GetIndex()

    def GetTrain(self, Batchsize, FoldIndex, shuffle=True,eval=True):
        trainindex = np.concatenate([self.index[:FoldIndex*self.testsize],self.index[(FoldIndex+1)*self.testsize:]])
        evalindex = trainindex[:self.testsize//2]
        trainindex = trainindex[self.testsize//2:]
        trainset = torch.utils.data.Subset(self.data, trainindex)
        evalset = torch.utils.data.Subset(self.data, evalindex)
        return torch.utils.data.DataLoader(trainset, batch_size=Batchsize, shuffle=shuffle), torch.utils.data.DataLoader(evalset, batch_size=Batchsize, shuffle=shuffle)
    
    def GetTest(self, Batchsize, FoldIndex, shuffle=False):
        if FoldIndex<4:
            testindex = self.index[FoldIndex*self.testsize:(FoldIndex+1)*self.testsize]
        else:
            testindex = self.index[FoldIndex*self.testsize:]
        testset = torch.utils.data.Subset(self.data, testindex)
        return torch.utils.data.DataLoader(testset, batch_size=Batchsize, shuffle=shuffle)

    def GetIndex(self):
        sample_by_folds = [[] for i in range(5)]
        sample_size_by_fold = 0
        for key in self.data.sites:
            sample_size = len(self.data.sites[key])
            index = np.random.permutation(np.array([i for i in range(sample_size)]))
            samples = self.data.sites[key]
            samples = list(np.array(samples)[index])
            for i in range(5):
                if i<4:
                    sample_by_folds[i] += samples[sample_size//5*i:sample_size//5*(i+1)]
                else:
                    sample_by_folds[i] += samples[sample_size//5*i:]
            sample_size_by_fold += sample_size//5
        sample_by_folds = np.concatenate(sample_by_folds)
        return sample_by_folds, sample_size_by_fold

    
def plot_contribution(ts,roi_pair,length):
    tsnorm = ((ts)/(torch.norm(ts,2,dim=0))).cpu().numpy()
    nts = (ts-ts.mean(-1,keepdim=True))/(ts.std(-1,keepdim=True)+1e-9)
    ts = ts/(ts.std(-1,keepdim=True)+1e-9)
    ts0, ts1 = ts[roi_pair[0],0:length].detach().cpu().numpy()/2, ts[roi_pair[1],0:length].detach().cpu().numpy()/2
    nts0, nts1 = nts[roi_pair[0],0:length].detach().cpu().numpy(), nts[roi_pair[1],0:length].detach().cpu().numpy()
    pos_ctb, neg_ctb, all_ctb = (nts0*nts1).copy()/2, (nts0*nts1).copy()/2, (nts0*nts1).copy()/2
    pos_ctb[pos_ctb<0] = 0
    neg_ctb[neg_ctb>0] = 0
    plt.xlabel('Time Point', fontsize=14)
    plt.ylabel('Signal Amplitude',fontsize=14)
    plt.plot(ts0,label=f'Singal Sequence of ROI {roi_pair[0]}',alpha=0.6,linewidth=1,marker='.',markersize=2.6)
    plt.plot(ts1,label=f'Singal Sequence of ROI {roi_pair[1]}',alpha=0.6,linewidth=1,marker='.',markersize=2.6)
    plt.bar(range(length),pos_ctb/2,alpha=0.5,color='green',edgecolor='black',label='Positive Contributions to PCC')
    plt.bar(range(length),neg_ctb/2,alpha=0.5,color='red',edgecolor='black',label='Negative Contributions to PCC')
    plt.xticks(np.arange(0, length+1, 15))
    plt.axhline(y=0,color="black", linestyle="--")
    # plt.axhline(y=-1,color="black", linestyle="--")
    # plt.axhline(y=-2,color="black", linestyle="--")

    plt.grid(True, which='both', axis='x', color='gray', linestyle='--', linewidth=0.5)

    plt.legend(fontsize=12)
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    expertize = torch.zeros(1,200)+0.01
    expertize[0,roi_pair] = 1
    expertize = expertize/torch.norm(expertize,2,dim=1)
    scores = torch.abs(expertize@tsnorm).flatten().detach().cpu().numpy()*4
    # plt.bar(range(length),scores,alpha=0.4,color='green',label='Positive Contribution')
    plt.show()

def plot_contribution_cos(ts,roi_pair,length):
    nts = ((ts)/(torch.norm(ts,2,dim=0))).cpu().numpy()
    expertize = torch.zeros(1,200)#+0.01
    expertize[0,roi_pair] = 1
    expertize = expertize/torch.norm(expertize,2,dim=1)
    scores = torch.abs(expertize@nts).flatten().detach().cpu().numpy()*10
    ts0, ts1 = ts[roi_pair[0],0:length].detach().cpu().numpy()/2, ts[roi_pair[1],0:length].detach().cpu().numpy()/2

    plt.plot(ts0+2,label='Singal Seuqnce 1')
    plt.plot(ts1+2,label='Singal Seuqnce 2')
    plt.bar(range(length),scores,alpha=0.5,color='blue',label='Positive Contribution')
    plt.axhline(y=2,color="black", linestyle="--")
    plt.axhline(y=0,color="black", linestyle="--")
    plt.show()

def accumulate(sequence):
    accum = np.array([sequence[0:i+1].sum() for i in range(len(sequence))])/6
    return accum


def coe(x,y,length):
    x = (x-x.mean(-1))/x.std(-1)
    y = (y-y.mean(-1))/y.std(-1)
    # plt.plot([i for i in range(x.shape[-1])],x-15)
    # plt.plot([i for i in range(x.shape[-1])],y-20)
    # plt.show()
    return (x*y).sum()/length