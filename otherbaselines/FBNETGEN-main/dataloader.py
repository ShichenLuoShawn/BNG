
import numpy as np
import torch, re, os, sys
import torch.utils.data as utils
from sklearn import preprocessing
import pandas as pd
from scipy.io import loadmat
import pathlib
import torch.nn.functional as F
from torch.utils.data import Dataset

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def infer_dataloader(dataset_config):

    label_df = pd.read_csv(dataset_config["label"])


    if dataset_config["dataset"] == "PNC":
        fc_data = np.load(dataset_config["time_seires"], allow_pickle=True).item()
        fc_timeseires = fc_data['data'].transpose((0, 2, 1))

        fc_id = fc_data['id']
    

        id2gender = dict(zip(label_df['SUBJID'], label_df['sex']))

        final_fc, final_label = [], []

        for fc, l in zip(fc_timeseires, fc_id):
            if l in id2gender:
                final_fc.append(fc)
                final_label.append(id2gender[l])
        final_fc = np.array(final_fc)


    elif dataset_config["dataset"] == 'ABCD':

        fc_data = np.load(dataset_config["time_seires"], allow_pickle=True)

    _, node_size, timeseries = final_fc.shape

    encoder = preprocessing.LabelEncoder()

    encoder.fit(label_df["sex"])

    labels = encoder.transform(final_label)

    final_fc = torch.from_numpy(final_fc).float()

    return final_fc, labels, node_size, timeseries


        
def init_dataloader(dataset_config,fold):

    if dataset_config["dataset"] == 'ABIDE':
        dataset = MyDatasetRaw("..\\..\\..\\Data\\"+'HCP_static','ts','cpu') #ADHD200
        # data = np.load(dataset_config["time_seires"], allow_pickle=True).item()
        timeseries, pearson, label = dataset.get_all()
        # print((label==1).sum(),(label==0).sum())
        # sys.exit()
        final_fc = timeseries
        final_pearson = pearson
        labels = label

    elif dataset_config["dataset"] == "HIV" or dataset_config["dataset"] == "BP":
        data = loadmat(dataset_config["node_feature"])

        labels = data['label']
        labels = labels.reshape(labels.shape[0])

        labels[labels==-1] = 0

        view = dataset_config["view"]

        final_pearson = data[view]

        final_pearson = np.array(final_pearson).transpose(2, 0, 1)

        final_fc = np.ones((final_pearson.shape[0],1,1))

    elif dataset_config["dataset"] == 'PPMI' or dataset_config["dataset"] == 'PPMI_balanced':
        m = loadmat(dataset_config["node_feature"])
        labels = m['label'] if dataset_config["dataset"] != 'PPMI_balanced' else m['label_new']
        labels = labels.reshape(labels.shape[0])
        data = m['X'] if dataset_config["dataset"] == 'PPMI' else m['X_new']
        final_pearson = np.zeros((data.shape[0], 84, 84))
        modal_index = 0
        for (index, sample) in enumerate(data):
            # Assign the first view in the three views of PPMI to a1
            final_pearson[index, :, :] = sample[0][:, :, modal_index]

        final_fc = np.ones((final_pearson.shape[0],1,1))

    else:

        fc_data = np.load(dataset_config["time_seires"], allow_pickle=True)
        pearson_data = np.load(dataset_config["node_feature"], allow_pickle=True)
        label_df = pd.read_csv(dataset_config["label"])

        if dataset_config["dataset"] == 'ABCD':

            with open(dataset_config["node_id"], 'r') as f:
                lines = f.readlines()
                pearson_id = [line[:-1] for line in lines]

            with open(dataset_config["seires_id"], 'r') as f:
                lines = f.readlines()
                fc_id = [line[:-1] for line in lines]

            id2pearson = dict(zip(pearson_id, pearson_data))

            id2gender = dict(zip(label_df['id'], label_df['sex']))

            final_fc, final_label, final_pearson = [], [], []

            for fc, l in zip(fc_data, fc_id):
                if l in id2gender and l in id2pearson:
                    if np.any(np.isnan(id2pearson[l])) == False:
                        final_fc.append(fc)
                        final_label.append(id2gender[l])
                        final_pearson.append(id2pearson[l])

            final_pearson = np.array(final_pearson)

            final_fc = np.array(final_fc)

        elif dataset_config["dataset"] == "PNC":
            pearson_data, fc_data = pearson_data.item(), fc_data.item()

            pearson_id = pearson_data['id']
            pearson_data = pearson_data['data']
            id2pearson = dict(zip(pearson_id, pearson_data))

            fc_id = fc_data['id']
            fc_data = fc_data['data']

            id2gender = dict(zip(label_df['SUBJID'], label_df['sex']))

            final_fc, final_label, final_pearson = [], [], []

            for fc, l in zip(fc_data, fc_id):
                if l in id2gender and l in id2pearson:
                    final_fc.append(fc)
                    final_label.append(id2gender[l])
                    final_pearson.append(id2pearson[l])

            final_pearson = np.array(final_pearson)

            final_fc = np.array(final_fc).transpose(0, 2, 1)

    _, _, timeseries = final_fc.shape

    _, node_size, node_feature_size = final_pearson.shape

    # scaler = StandardScaler(mean=np.mean(
    #     final_fc), std=np.std(final_fc))
    
    # final_fc = scaler.transform(final_fc)

    if dataset_config["dataset"] == 'PNC' or dataset_config["dataset"] == 'ABCD':

        encoder = preprocessing.LabelEncoder()

        encoder.fit(label_df["sex"])

        labels = encoder.transform(final_label)
        

    # final_fc, final_pearson, labels = [torch.from_numpy(
    #     data).float() for data in (final_fc, final_pearson, labels)]

    length = final_fc.shape[0]
    train_length = int(length*0.8)
    val_length = int(1)

    index = np.random.permutation(np.array([i for i in range(length)]))
    testsize = length//5
    trainindex = np.concatenate([index[:fold*testsize],index[(fold+1)*testsize:]])
    testindex = index[fold*testsize:(fold+1)*testsize]
        
    dataset = utils.TensorDataset(
        final_fc,
        final_pearson,
        labels
    )
    trainset = torch.utils.data.Subset(dataset, trainindex)
    testset = torch.utils.data.Subset(dataset, testindex)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_length, 1, length-train_length-val_length])

    train_dataloader = utils.DataLoader(
        trainset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

    test_dataloader = utils.DataLoader(
        testset, batch_size=dataset_config["batch_size"], shuffle=False, drop_last=False)

    return (train_dataloader, val_dataloader, test_dataloader), node_size, node_feature_size, timeseries



class MyDatasetRaw(Dataset):
    def __init__(self,path,option='ts',device='cpu'):
        self.device=device
        self.op = option
        self.timeSeries = []
        self.pearson = []
        self.labels = []
        self.length = []
        self.path = path
        self.metadata = pd.read_csv(f'{path}\\'+'label.csv')[["subject","DX_GROUP"]] if path[-6:] != 'static' else pd.read_csv(f'{path}\\'+'label.csv')[['Subject','Gender']]
        self.LoadData(pearson=False)
        # print(np.unique(np.array(self.length)))
        # sys.exit()
        
    def __len__(self):
        return len(self.timeSeries)
    
    def __getitem__(self, index):
        return self.timeSeries[index],self.labels[index],self.length[index]
    
    def LoadData(self, pearson=False, top25=True):
        if not pearson:
            for root, dirs, files in os.walk(self.path):
                for file in files:
                    if file.endswith('cc200.1D'):
                        timeSeries = torch.Tensor(np.loadtxt(root+'\\'+file,dtype=float).transpose(1,0)).to(self.device)
                        length = timeSeries.shape[1]
                        if length<100:
                            continue
                        length2 = 116
                        timeSeries = timeSeries[:,0:116]
                        timeSeries = (timeSeries)/(timeSeries.std(-1,keepdim=True)+1e-9)
                        ID = file[-19:-14]
                        label = self.metadata.loc[self.metadata['subject']==int(ID)].iloc[0,1]-1 # 1 autism 0 control
                    elif file.endswith('cc200.csv'):
                        ID = int(re.search('^[0-9]*_',file).group()[0:-1])
                        label = self.metadata.loc[self.metadata['subject']==int(ID)].iloc[0,1]
                        if label == 'withheld': continue
                        else: label = int(label)
                        label = 1 if label>=1 else 0

                        timeSeries = torch.Tensor(np.array(pd.read_csv(root+'\\'+file,dtype=float)).transpose(1,0)).to(self.device)[1:,:]
                        length = timeSeries.shape[1]
                        if length<100:
                            continue
                        length2 = 120
                        timeSeries = timeSeries[:,0:120]
                        timeSeries = (timeSeries-timeSeries.mean(-1,keepdim=True))/(timeSeries.std(-1,keepdim=True)+1e-9)

                    elif file.endswith('.txt'):
                        ID = file[:6]
                        label = self.metadata.loc[self.metadata['Subject']==int(ID)].iloc[0,1]
                        label = 1 if label=='M' else 0
                        timeSeries = torch.Tensor(np.loadtxt(root+'\\'+file,dtype=float)).to(self.device).transpose(1,0)
                        length = timeSeries.shape[-1]
                        length2 = timeSeries.shape[-1]
                        paddingsize = 4000-length
                        timeSeries = (timeSeries-timeSeries.mean(-1,keepdim=True))/(timeSeries.std(-1,keepdim=True)+1e-9)
                        timeSeries = F.pad(timeSeries,(0,paddingsize),'constant',0)

                    self.pearson.append((timeSeries@timeSeries.T/length2)[None,:,:])
                    self.timeSeries.append(timeSeries[None,:,:])
                    self.labels.append(label)
                    self.length.append(length)

            self.labels=torch.Tensor(self.labels).to(self.device)
            self.timeSeries=torch.cat(self.timeSeries).to(self.device)
            self.pearson = torch.cat(self.pearson).to(self.device)

        else:
            for root, dirs, files in os.walk(self.path):
                for file in files:
                    if not file.endswith('cc200.1D'):continue
                    timeSeries = torch.Tensor(np.loadtxt(root+'\\'+file,dtype=float).transpose(1,0)).to(self.device)
                    length = timeSeries.shape[1]

                    timeSeries = (timeSeries-timeSeries.mean(-1,keepdim=True))/(timeSeries.std(-1,keepdim=True)+1e-9)
                    if top25:
                        # median1 = torch.abs(timeSeries).median()
                        k = int(length*0.75)
                        value,indice = torch.topk(torch.abs(timeSeries),k)
                        mask = torch.zeros(timeSeries.shape,device=self.device)
                        mask[torch.arange(200)[:,None],indice] = 1
                        # mask =  1-mask
                        timeSeries = timeSeries*mask
                    timeSeries = (timeSeries-timeSeries.mean(-1,keepdim=True))/(timeSeries.std(-1,keepdim=True)+1e-9)
                    # timeSeries = F.pad(timeSeries,(0,paddingsize),'constant',0)[None,:,:]
                    pc = timeSeries@timeSeries.T/(length+1)
                    ID = file[-19:-14]
                    label = self.metadata.loc[self.metadata['subject']==int(ID)].iloc[0,1]-1
                    self.timeSeries.append(pc)
                    self.labels.append(label)
                    self.length.append(length)

    def get_all(self):
        return self.timeSeries,self.pearson,self.labels