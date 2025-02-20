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

class MyDatasetRaw(Dataset):
    def __init__(self,path,option='ts',device='cpu'):
        self.device=device
        self.op = option
        self.timeSeries = []
        self.labels = []
        self.length = []
        self.path = path
        self.metadata = pd.read_csv(f'{path}\\'+'label.csv')[["subject","DX_GROUP"]] if path[-6:] != 'static' else pd.read_csv(f'{path}\\'+'label.csv')[['Subject','Gender']]
        self.LoadData(pearson=False)
        
    def __len__(self):
        return len(self.timeSeries)
    
    def __getitem__(self, index):
        return self.timeSeries[index],self.labels[index],self.length[index]
    
    def LoadData(self, pearson=False, top25=True):
        if not pearson:
            for root, dirs, files in os.walk(self.path):
                for file in files:
                    if file.endswith('cc200.1D'): #load ABIDE
                        timeSeries = torch.Tensor(np.loadtxt(root+'\\'+file,dtype=float).transpose(1,0)).to(self.device)
                        length = timeSeries.shape[1]
                        paddingsize = 360-length
                        timeSeries = (timeSeries-timeSeries.mean(-1,keepdim=True))/(timeSeries.std(-1,keepdim=True)+1e-9)
                        timeSeries = F.pad(timeSeries,(0,paddingsize),'constant',0)[None,:,:]
                        ID = file[-19:-14]
                        label = self.metadata.loc[self.metadata['subject']==int(ID)].iloc[0,1]-1 # 1 autism 0 control
                    elif file.endswith('cc200.csv'): #Load_ADHD
                        ID = int(re.search('^[0-9]*_',file).group()[0:-1])
                        label = self.metadata.loc[self.metadata['subject']==int(ID)].iloc[0,1]
                        if label == 'withheld': continue
                        else: label = int(label)
                        label = 1 if label>=1 else 0
                        timeSeries = torch.Tensor(np.array(pd.read_csv(root+'\\'+file,dtype=float)).transpose(1,0)).to(self.device)[1:,:]
                        length = timeSeries.shape[1]
                        paddingsize = 400-length
                        timeSeries = (timeSeries-timeSeries.mean(-1,keepdim=True))/(timeSeries.std(-1,keepdim=True)+1e-9)
                        timeSeries = F.pad(timeSeries,(0,paddingsize),'constant',0)[None,:,:]
                    elif file.endswith('.txt'): #Load HCP
                        ID = file[:6]
                        label = self.metadata.loc[self.metadata['Subject']==int(ID)].iloc[0,1]
                        label = 1 if label=='M' else 0
                        timeSeries = torch.Tensor(np.loadtxt(root+'\\'+file,dtype=float)).to(self.device).transpose(1,0)
                        length = timeSeries.shape[-1]
                        timeSeries = ((timeSeries-timeSeries.mean(-1,keepdim=True))/(timeSeries.std(-1,keepdim=True)+1e-9))[None,:,:]

                    if self.op=='pc':
                        self.timeSeries.append(timeSeries@timeSeries.transpose(2,1)/length) #Pearson Correlation
                    else:
                        self.timeSeries.append(timeSeries)
                    self.labels.append(label)
                    self.length.append(length)

            self.labels=torch.Tensor(self.labels).to(self.device)
            self.timeSeries=torch.cat(self.timeSeries).to(self.device)

        else:
            for root, dirs, files in os.walk(self.path):
                for file in files:
                    if not file.endswith('cc200.1D'):continue
                    timeSeries = torch.Tensor(np.loadtxt(root+'\\'+file,dtype=float).transpose(1,0)).to(self.device)
                    length = timeSeries.shape[1]

                    timeSeries = (timeSeries-timeSeries.mean(-1,keepdim=True))/(timeSeries.std(-1,keepdim=True)+1e-9)
                    if top25:
                        k = int(length*0.75)
                        value,indice = torch.topk(torch.abs(timeSeries),k)
                        mask = torch.zeros(timeSeries.shape,device=self.device)
                        mask[torch.arange(200)[:,None],indice] = 1
                        timeSeries = timeSeries*mask
                    timeSeries = (timeSeries-timeSeries.mean(-1,keepdim=True))/(timeSeries.std(-1,keepdim=True)+1e-9)
                    pc = timeSeries@timeSeries.T/(length+1)
                    ID = file[-19:-14]
                    label = self.metadata.loc[self.metadata['subject']==int(ID)].iloc[0,1]-1
                    self.timeSeries.append(pc)
                    self.labels.append(label)
                    self.length.append(length)
    
class DataSpliter():
    def __init__(self,data,NumofFold:int,testRate=1,RSeed = 0):
        assert (testRate<=1 and testRate>0), 'testRate must within (0,1]'
        assert type(NumofFold)==int, 'NumofFold must be an integer'
        self.Rseed = RSeed
        self.data = data
        self.testRate = testRate
        self.NumofFold = NumofFold
        self.index, self.testsize = self.GetIndex()

    def GetTest(self, Batchsize, FoldIndex, shuffle=False):
        testindex = self.index[FoldIndex*self.testsize:(FoldIndex+1)*self.testsize]
        testset = torch.utils.data.Subset(self.data, testindex)
        return torch.utils.data.DataLoader(testset, batch_size=Batchsize, shuffle=shuffle)

    def GetTrain(self, Batchsize, FoldIndex, shuffle=True, eval=True):
        if not eval:
            trainindex = np.concatenate([self.index[:FoldIndex*self.testsize],self.index[(FoldIndex+1)*self.testsize:]])
            trainset = torch.utils.data.Subset(self.data, trainindex)
            return torch.utils.data.DataLoader(trainset, batch_size=Batchsize, shuffle=shuffle), 0
        else:
            trainindex = np.concatenate([self.index[:FoldIndex*self.testsize],self.index[(FoldIndex+1)*self.testsize:]])
            evalindex = trainindex[:self.testsize]
            trainindex = trainindex[self.testsize:]
            trainset = torch.utils.data.Subset(self.data, trainindex)
            evalset = torch.utils.data.Subset(self.data, evalindex)
            return torch.utils.data.DataLoader(trainset, batch_size=Batchsize, shuffle=shuffle), torch.utils.data.DataLoader(evalset, batch_size=Batchsize, shuffle=shuffle)

    def GetIndex(self):
        DataLength = len(self.data)
        index = np.random.permutation(np.array([i for i in range(DataLength)]))
        if self.testRate==1:
            testsize = DataLength//self.NumofFold
        else:
            testsize = int(DataLength//self.NumofFold*self.testRate)
        return index, testsize
    

if __name__=='__main__':
    #dataset = MyDataset('ABIDE.mat',False)
    #dataset=node_embedding('AAL90_region_info.xls')
    pass


