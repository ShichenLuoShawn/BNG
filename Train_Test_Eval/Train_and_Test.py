import torch,math
from Train_Test_Eval import Evaluate
import sys, os, json, argparse, random
from models.stagin_main.experiment import step
import models.stagin_main.util as util
from einops import repeat
import torch.nn.functional as F

def train(dataloader,evalloader, model, optimizer, lossfunc, epoch, trainevaluator,evalevaluator\
         ,args):
    model.train()
    trainevaluator.reset(epoch)
    if args.model_name!='stagin':
        for i,(inptrain,label,length) in enumerate(dataloader):
            inp, label, length = inptrain.to(args.device), label.long().to(args.device), length.to(args.device)
            optimizer.zero_grad()
            use_id = True if args.spec=='id' else False
            out, top_k_loss, Relaxed_ortho, rec = model(inp,length,'train',epoch, args, label)
            claLoss = lossfunc['CE'](torch.Tensor(out), label)
            loss = claLoss + args.alpha*top_k_loss + Relaxed_ortho + rec #rec is for autoencoder as classifier only
            trainevaluator.update(out,label,[claLoss.item(), top_k_loss.item(),Relaxed_ortho.item()])
            loss.backward()
            optimizer.step()

    else:
        for i,(inptrain,label,length) in enumerate(dataloader):
            inp, label, length = inptrain.to(args.device), label.long().to(args.device), length.to(args.device)
            optimizer.zero_grad()
            dyn_a, sampling_points = util.bold.process_dynamic_fc(inp.transpose(2,1), 75, 35, None)
            sampling_endpoints = [p+50 for p in sampling_points]
            if i==0: dyn_v = repeat(torch.eye(args.node_size), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=inp.shape[0])
            if len(dyn_a) < args.batch_size: dyn_v = dyn_v[:len(dyn_a)]
            t = inp.transpose(2,1).permute(1,0,2)

            logit, attention, latent, reg_ortho = model(dyn_v.to(args.device), dyn_a.to(args.device), t.to(args.device), sampling_endpoints)
            claLoss = lossfunc['CE'](torch.Tensor(logit), label)
            loss = claLoss + 0.00001*reg_ortho
            trainevaluator.update(logit,label,[claLoss.item(), reg_ortho.item(), torch.tensor(0)])
            loss.backward()
            optimizer.step()
    return False

def test(dataloader, model, testevaluator, epoch, fold, args):
    model.eval()
    testevaluator.reset(0)
    if args.model_name!='stagin':
        for i,(inptest,label,length) in enumerate(dataloader):
            use_id = True if args.spec=='id' else False
            inp, label, length = inptest.to(args.device), label.to(args.device), length.to(args.device)
            out, top_k_loss, Relaxed_ortho, rec = model(inp,length,'test',0,args,label)
            testevaluator.update(out,label,0)
    else:
        for i,(inptest,label,length) in enumerate(dataloader):
            inp, label, length = inptest.to(args.device), label.to(args.device), length.to(args.device)
            dyn_a, sampling_points = util.bold.process_dynamic_fc(inp.transpose(2,1), 50, args.batch_size, None)
            sampling_endpoints = [p+50 for p in sampling_points]
            if i==0: dyn_v = repeat(torch.eye(args.node_size), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=inp.shape[0])
            if len(dyn_a) < args.batch_size: dyn_v = dyn_v[:len(dyn_a)]
            t = inp.transpose(2,1).permute(1,0,2)

            logit, attention, latent, reg_ortho = model(dyn_v.to(args.device), dyn_a.to(args.device), t.to(args.device), sampling_endpoints)
            testevaluator.update(logit,label,0)

def save(fold, model, testevaluator, seed, args, content='model'):
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    record_file_name = f'data={args.data_name}_model={args.classifier}_spec={args.spec}_seed={seed}'
    model_name = f'data={args.data_name}_model={args.classifier}_spec={args.spec}.pth'
    sub_finalpath = f'{args.classifier}'
    sub_finalpath = os.path.join(args.save_path, sub_finalpath)
    sub_finalpath = os.path.join(sub_finalpath, f'{args.data_name}_top{args.topk}_{args.spec}')
    if not os.path.isdir(sub_finalpath):
        os.makedirs(sub_finalpath)
    
    pathformodel = os.path.join(sub_finalpath, model_name)
    if content == 'model':
        print('saving model......')
        torch.save(model.state_dict(),pathformodel)
    sub_finalpath = os.path.join(sub_finalpath, record_file_name)
    if content == 'metric':
        print('saving metrics......')
        testevaluator.save_result(sub_finalpath)

