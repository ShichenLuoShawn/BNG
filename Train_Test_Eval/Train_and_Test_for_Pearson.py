import torch
from Train_Test_Eval import Evaluate
import sys, os, json, argparse, random

def train(dataloader,evalloader, model, optimizer, lossfunc, epoch, trainevaluator,evalevaluator\
         ,args):
    model.train()
    trainevaluator.reset(epoch)
    for i,(inptrain,label,length) in enumerate(dataloader):
        inp, label, length = inptrain.to(args.device), label.long().to(args.device), length.to(args.device)
        optimizer.zero_grad()
        if args.model_name=='aePC':
            out = model(inp)
        else:
            out = model(inp,inp)
        loss = lossfunc['CE'](torch.Tensor(out), label)
        trainevaluator.update(out,label,[loss.item()])
        loss.backward()
        optimizer.step()
    if (epoch%args.test_interval==0 or epoch==2) and evalloader:
        model.eval()
        evalevaluator.reset(0)
        for _, (inptrain, label) in enumerate(evalloader):
            inp, label = inptrain.to(args.device), label.long().to(args.device)
            out = model(inp)
            evalevaluator.update(out,label,0)
    return False

def test(dataloader, model, testevaluator, epoch, fold, args):
    model.eval()
    testevaluator.reset(0)
    for i,(inptest,label,length) in enumerate(dataloader):
        inp, label, length = inptest.to(args.device), label.to(args.device), length.to(args.device)
        if args.model_name=='aePC':
            out = model(inp)
        else:
            out = model(inp,inp)
        testevaluator.update(out,label,0)

def save(fold, model, testevaluator, seed, args, content='model'):
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    record_file_name = f'data={args.data_name}_model={args.classifier}_spec={args.spec}_seed={seed}'
    sub_finalpath = f'{args.classifier}'
    sub_finalpath = os.path.join(args.save_path, sub_finalpath)
    sub_finalpath = os.path.join(sub_finalpath, f'{args.data_name}_top{args.topk}_{args.spec}')
    if not os.path.isdir(sub_finalpath):
        os.makedirs(sub_finalpath)

    sub_finalpath = os.path.join(sub_finalpath, record_file_name)
    if content == 'model':
        print('saving model......')
        torch.save(model.state_dict(),sub_finalpath)
        with open(f'{sub_finalpath}/args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    elif content == 'metric':
        print('saving metrics......')
        testevaluator.save_result(sub_finalpath)

def load_args(path):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(path, 'r') as f:
        args.__dict__ = json.load(f)


