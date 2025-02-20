import torch, time
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os,sys,random
from data_loader import Load_Data
from Train_Test_Eval import Evaluate, Train_and_Test
from models.KPFBNC import KPFBNC, weight_init
import numpy as np
import torch_geometric
import torch.optim.lr_scheduler as lr_scheduler
import argparse, json
from models.stagin_main.model import ModelSTAGIN
from models.BNT.BNT.bnt import BrainNetworkTransformer
from models.Baselines import GCN, AE


parser = argparse.ArgumentParser(
                    prog='Key Pattern Focused Brain Network Construction')
#data options
parser.add_argument('--data_name', type=str, default='ABIDE200') #HCP_static  ABIDE200 ADHD200
parser.add_argument('--node_size', type=int, default=200) 
parser.add_argument('--data_type', type=str, default='time_series')
parser.add_argument('--num_of_fold', type=int, default=5)

#training options
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--model_name', type=str, default='KPFBGC') # KPFBGC stagin  BNT GCN AE
parser.add_argument('--classifier', type=str, default='ae')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_of_epoch', type=int, default=100)
parser.add_argument('--random_seed', type=list, default=[i for i in range(5)])

#model hyperparameters
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--topk', type=int, default=4)
parser.add_argument('--alpha', type=float, default=0.25)
parser.add_argument('--stride', type=int, default=25)
parser.add_argument('--threshold', type=int, default=0.5)
parser.add_argument('--num_pattern', type=int, default=400)

#evaluate options
parser.add_argument('--test_interval', type=int, default=30)
parser.add_argument('--save_model', type=bool, default=False)
parser.add_argument('--save_result', type=bool, default=True)
parser.add_argument('--save_path', type=str, default='resultRecord2/')
parser.add_argument('--spec', type=str, default='')

parser.add_argument('--compare', type=bool, default=True)

   
args = parser.parse_args()

for classifier in ['ae','BNT','gcn']:#'ae','BNT' 'gcn'
    args.lr = 1e-3
    args.classifier = classifier
    for thres in [0.5]:
        args.threshold = thres
        args.spec = f'thres_{thres}'
        if args.data_type == 'time_series':
            if args.data_name == 'ABIDE200full' or args.data_name == 'HCP_static':
                dataset0 = Load_Data.MyDatasetRaw("..\\Data\\"+args.data_name,'ts','cuda')
            else:
                dataset0 = Load_Data.MyDatasetRaw("TimeSeries\\"+args.data_name,'ts','cuda')#args.device)
        for seed in args.random_seed:
            np.random.seed(seed)
            torch.random.seed=seed
            torch.cuda.manual_seed_all(seed)
            torch.manual_seed(seed) 
            random.seed(seed)
            dataset = Load_Data.DataSpliter(dataset0,args.num_of_fold)
            lossfunc = {'CE':nn.CrossEntropyLoss(),'MSE':nn.MSELoss()}
            Trainevaluator = Evaluate.Evaluator()
            Evalevaluator = Evaluate.Evaluator()
            Evalevaluator2= Evaluate.Evaluator()
            Testevaluator = Evaluate.Evaluator()
            Eval = False

            for fold in range(1,args.num_of_fold+1):
                Models = {'ae':AE(args.node_size),\
                        'KPFBGC':KPFBNC(args.node_size,topK=args.topk,stride=args.stride,classifier=args.classifier,use_id=None,num_of_pattern=400),\
                        'gcn':GCN(),\
                        'BNT':BrainNetworkTransformer(args.node_size,200//args.topk),\
                        'stagin':ModelSTAGIN(input_dim=200, hidden_dim=128, num_classes=2, num_heads=1, \
                        num_layers=4, sparsity=30, dropout=0.5, cls_token='sum', readout='sero',)}
                t1 = time.perf_counter()
                trainloader, evalloader = dataset.GetTrain(Batchsize=args.batch_size,FoldIndex=fold-1,eval=Eval)
                testloader = dataset.GetTest(Batchsize=args.batch_size,FoldIndex=fold-1)
                model = Models[args.model_name]
                weight_init(model)
                model.to(args.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

                e, repeated_time, max_repreat_time,repeated_time2 = 1, 0, 5, 0 #some model need more training epoch to converge
                while e < args.num_of_epoch+1: 
                    stop = Train_and_Test.train(trainloader,evalloader,model,optimizer,lossfunc,e,Trainevaluator,Evalevaluator,args)
                    if (e)%args.test_interval==0 or (e)==args.num_of_epoch:# or e==1:
                        Train_and_Test.test(testloader,model,Testevaluator,e,fold,args)
                        if Eval:  
                            print('epo:', e, "test acc:", round(Testevaluator.accuracy(),4), "val acc:", round(Evalevaluator.accuracy(),4),\
                            'train acc:', round(Trainevaluator.accuracy(),4), 'auc',round(Testevaluator.AUC(),4),'loss', Trainevaluator.losses[0],Trainevaluator.losses[1])
                        else:
                            print('epo:', e, "test acc:", round(Testevaluator.accuracy(),4),'auc:',Testevaluator.AUC(),'pre:',Testevaluator.precision(),\
                                'rec:',Testevaluator.recall(), 'sen:', Testevaluator.sensitivity(), 'spe:', Testevaluator.specificity(),\
                                    'train acc:', round(Trainevaluator.accuracy(),4),'loss:', Trainevaluator.losses[0], Trainevaluator.losses[1],Trainevaluator.losses[2])
                    if e==args.num_of_epoch and (Trainevaluator.accuracy()<=1-1e-9 or Trainevaluator.losses[0]>=1e-2) and repeated_time<max_repreat_time :
                        e-=40
                        repeated_time+=1
                    if e==40 and Trainevaluator.accuracy()<=0.65 and repeated_time2<1:
                        e-=30
                        weight_init(model)
                        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/2,weight_decay=args.weight_decay)
                        repeated_time2+=1
                    e+=1
                Testevaluator.accuracy(record=True)
                Testevaluator.recall(record=True)
                Testevaluator.precision(record=True)
                Testevaluator.AUC(record=True)
                Testevaluator.sensitivity(record=True)
                Testevaluator.specificity(record=True)

                print('k:',args.topk,'seed:',seed,'fold:',fold,'N-fold mean:',Testevaluator.foldAVG('acc'),'all_acc',Testevaluator.record['acc'],'auc:',Testevaluator.foldAVG('auc'),)
                t2 = time.perf_counter()
                print(f'time comsumed: {t2-t1}\n')
                if args.save_model:
                    Train_and_Test.save(fold,model,Testevaluator,seed, args,'model')
            if args.save_result:
                Train_and_Test.save(fold,model,Testevaluator,seed,args,'metric') #metric
            print(Testevaluator.foldAVG('acc'))


        if args.compare:
            Evaluate.compare(args)