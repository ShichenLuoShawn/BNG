import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
import pandas as pd
import os

class Evaluator():
    def __init__(self):
        self.fold = 1
        self.record = {'acc':[],'pre':[],'rec':[],'sen':[],'spe':[],'auc':[]} # {epoch:acc}
        self.temptp = 0
        self.tempfp = 0
        self.temptn = 0
        self.tempfn = 0
        self.loss = 0
        self.losses=[]
        self.preds = np.array([])
        self.labels = np.array([])

    def update(self, out, groundtruth,loss):
        self.labels = np.concatenate([self.labels,groundtruth.flatten().to('cpu')],0)
        self.preds = np.concatenate([self.preds,F.softmax(out,-1)[:,1].flatten().detach().to('cpu')],0)
        pred = torch.argmax(out,-1)
        self.temptp += ((pred==1)*(groundtruth==1)).sum() #true positive
        self.tempfp += ((pred==1)*(groundtruth==0)).sum()
        self.temptn += ((pred==0)*(groundtruth==0)).sum()
        self.tempfn += ((pred==0)*(groundtruth==1)).sum()
        self.losses = loss
        if loss:
            self.loss = torch.tensor(loss).sum().item()

    def accuracy(self,record=False):
        acc = round(((self.temptp+self.temptn)/(self.temptp+self.tempfp+self.temptn+self.tempfn)).item(),4)
        if record:
            self.record['acc'].append(acc)
        return acc
    
    def precision(self,record=False):
        pre = round((self.temptp/(self.temptp+self.tempfp)).item(),4)
        if record:
            self.record['pre'].append(pre)
        return pre
    
    def recall(self,record=False):
        rec = round((self.temptp/(self.temptp+self.tempfn)).item(),4)
        if record:
            self.record['rec'].append(rec)
        return rec
    
    def sensitivity(self,record=False):
        sen = round((self.temptp/(self.temptp+self.tempfn)).item(),4)
        if record:
            self.record['sen'].append(sen)
        return sen
    
    def specificity(self,record=False):
        spe = round((self.temptn/(self.temptn+self.tempfp)).item(),4)
        if record:
            self.record['spe'].append(spe)
        return spe
    
    def AUC(self,record=False):
        auc = round(metrics.roc_auc_score(self.labels,self.preds),4)
        if record:
            self.record['auc'].append(auc)
        return auc
    
    def loss(self,record=False):
        if record:
            self.record['loss'].append(self.losses)
        return self.losses
    
    def foldAVG(self,metrics):
        return round(torch.tensor(self.record[metrics]).mean().item(),4)

    def reset(self, fold):
        self.temptp = 0
        self.tempfp = 0
        self.temptn = 0
        self.tempfn = 0
        self.loss=0
        self.fold = fold
        self.preds = np.array([])
        self.labels = np.array([])
    
    def save_result(self, path_name,record_mean_std=True):
        if record_mean_std:
            for key in self.record.keys():
                data = self.record[key]
                mean = np.array(data).mean()
                std = np.array(data).std()
                self.record[key].append(mean)
                self.record[key].append(std)
        data = pd.DataFrame(self.record)
        data.index = [1,2,3,4,5,'mean','std']
        data.to_csv(f'{path_name}.csv')

def compare(args):
    record = {'acc':[],'pre':[],'rec':[],'sen':[],'spe':[],'auc':[]}
    record2 = {'acc':[],'pre':[],'rec':[],'sen':[],'spe':[],'auc':[]}
    for seed in args.random_seed:
        record_file_name = f'data={args.data_name}_model={args.classifier}_spec={args.spec}_seed={seed}.csv'
        subpath2 = f'./resultRecord/{args.classifier}/{args.data_name}_top{args.topk}_{args.spec}'
        path2 = os.path.join(subpath2,record_file_name)
        res2 = pd.read_csv(path2)
        res2 = res2.set_index('Unnamed: 0')
        mean2 = res2.loc['mean']
        for key in record2.keys():
            record2[key].append(mean2[key])
    for key in record.keys():
        record2[key].append(np.array(record2[key]).mean())
    data2 = pd.DataFrame(record2)
    data2.to_csv(f'{subpath2}/result.csv')
