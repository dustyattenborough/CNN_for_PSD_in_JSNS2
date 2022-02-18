#!/usr/bin/env python
import sys, os
import argparse

import numpy as np
import csv, yaml

import h5py
import torch
import torch.nn as nn

sys.path.append("./python")

from dataset.WFCh2Dataset_each_norm import *
from dataset.WFCh2Dataset_max_norm import *



from models.allModels import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', action='store', type=str, default='config.yaml', help='Configration file with sample information')
parser.add_argument('--data', action='store', type=str, help='data type select')
parser.add_argument('-o', '--output', action='store', type=str, required=True, help='Path to output directory')
parser.add_argument('--norm', action='store', type=int, default = 0, help='max norm =0 / each norm = 1')
parser.add_argument('--device', action='store', type=int, default=0, help='device name')
parser.add_argument('--epoch', action='store', type=int, help='Number of epochs')
parser.add_argument('--batch', action='store', type=int, default=32, help='Batch size')
parser.add_argument('--lr', action='store', type=float, help='Learning rate')
parser.add_argument('--seed', action='store', type=int, help='random seed')
parser.add_argument('--kernel_size', action='store', type=int, default=3, help='kernel size at the 1st layer')
parser.add_argument('--odd', action='store', type=float, help='odd =1 or total = 0')
parser.add_argument('--runnum', action='store', type=int, help='runnumber')
parser.add_argument('--rho', action='store', type=float, help='rho')
parser.add_argument('--vtz', action='store', type=float, help='vtz')
parser.add_argument('--minvalue', action='store', type=int, default = 400, help='min cut')
parser.add_argument('--dvertex', action='store', type=int, default = 1, help='no dvertex cut =0 / dvertex cut = 1')


models = ['1DCNN3FC2', '1DCNN3FC2log']
parser.add_argument('--model', choices=models, default=models[0], help='model name')
args = parser.parse_args()

config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)
config['training']['learningRate'] = float(config['training']['learningRate'])
if args.seed: config['training']['randomSeed1'] = args.seed
if args.epoch: config['training']['epoch'] = args.epoch
if args.lr: config['training']['learningRate'] = args.lr

torch.set_num_threads(os.cpu_count())
if torch.cuda.is_available() and args.device >= 0: torch.cuda.set_device(args.device)
if not os.path.exists('result/' + args.output): os.makedirs('result/' + args.output)

##### Define dataset instance #####
if args.norm == 0:
    dset = WFCh2Dataset_each_norm(channel=config['format']['channel'], output = 'result/' + args.output)
else:
    dset = WFCh2Dataset_max_norm(channel=config['format']['channel'], output = 'result/' + args.output)
    
    
for sampleInfo in config['samples']:
    if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
    
    name = sampleInfo['name']
    
    if args.odd == 1:
        in_path = 'com_data/r00'+str(args.runnum) + '_' + str(args.data) +'/'+name+'_even_Rho_'+str(args.rho)+'_ZL_'+str(args.vtz)+'_min_'+str(args.minvalue)+'_dv_'+str(args.dvertex)+'/*.h5'
    elif args.odd == 2:
        in_path = 'com_data/r00'+str(args.runnum) + '_' + str(args.data) +'/'+name+'_odd_Rho_'+str(args.rho)+'_ZL_'+str(args.vtz)+'_min_'+str(args.minvalue)+'_dv_'+str(args.dvertex)+'/*.h5'
    
    else: ## total training
        in_path = 'com_data/r00'+str(args.runnum) + '_' + str(args.data) +'/'+name+'_cut_Rho_'+str(args.rho)+'_ZL_'+str(args.vtz)+'_min_'+str(args.minvalue)+'_dv_'+str(args.dvertex)+'/*.h5'

    dset.addSample(name, in_path, weight=sampleInfo['xsec']/sampleInfo['ngen'])
    dset.setProcessLabel(name, sampleInfo['label'])
dset.initialize()

lengths = [int(x*len(dset)) for x in config['training']['splitFractions']]
lengths.append(len(dset)-sum(lengths))
torch.manual_seed(config['training']['randomSeed1'])
trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)

kwargs = {'num_workers':min(config['training']['nDataLoaders'], os.cpu_count()), 'pin_memory':False}
from torch.utils.data import DataLoader
trnLoader = DataLoader(trnDset, batch_size=args.batch, shuffle=True, **kwargs)
valLoader = DataLoader(valDset, batch_size=args.batch, shuffle=False, **kwargs)
torch.manual_seed(torch.initial_seed())


##### Define model instance #####
exec('model = WF'+args.model+'Model(nChannel=dset.channel, nPoint=dset.width, kernel_size=args.kernel_size, device=args.device)')
torch.save(model, os.path.join('result/' + args.output, 'model.pth'))

device = 'cpu'
if args.device >= 0 and torch.cuda.is_available():
    model = model.cuda()
    device = 'cuda'

##### Define optimizer instance #####
optm = torch.optim.Adam(model.parameters(), lr=config['training']['learningRate'])

##### Start training #####
with open('result/' + args.output+'/summary.txt', 'w') as fout:
    fout.write(str(args))
    fout.write('\n\n')
    fout.write(str(model))
    fout.close()

from sklearn.metrics import accuracy_score
from tqdm import tqdm
bestState, bestLoss = {}, 1e9
train = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
nEpoch = config['training']['epoch']
for epoch in range(nEpoch):
    model.train()
    trn_loss, trn_acc = 0., 0.
    nProcessed = 0
    optm.zero_grad()
    for i, (data, label0, weight, rescale, procIdx, fileIdx, idx) in enumerate(tqdm(trnLoader, desc='epoch %d/%d' % (epoch+1, nEpoch))):
        data = data.to(device)

        label = label0.float().to(device)
        weight = (weight*rescale).float().to(device)

        pred = model(data)
        crit = torch.nn.BCEWithLogitsLoss(weight=weight)
        loss = crit(pred.view(-1), label)
        loss.backward()

        optm.step()
        optm.zero_grad()

        label0 = label0.reshape(-1).numpy()
        ibatch = len(label0)
        nProcessed += ibatch
        trn_loss += loss.item()*ibatch
        trn_acc += accuracy_score(label0, np.where(pred.to('cpu') > 0.5, 1, 0), sample_weight=weight.to('cpu'))*ibatch
    trn_loss /= nProcessed 
    trn_acc  /= nProcessed

    model.eval()
    val_loss, val_acc = 0., 0.
    nProcessed = 0
    for i, (data, label0, weight, rescale, procIdx, fileIdx, idx) in enumerate(tqdm(valLoader)):
        data = data.to(device)
        label = label0.float().to(device)
        weight = (weight*rescale).float().to(device)

        pred = model(data)
        crit = torch.nn.BCEWithLogitsLoss(weight=weight)
        loss = crit(pred.view(-1), label)

        label0 = label0.reshape(-1).numpy()
        ibatch = len(label0)
        nProcessed += ibatch
        val_loss += loss.item()*ibatch
        val_acc += accuracy_score(label0, np.where(pred.to('cpu') > 0.5, 1, 0), sample_weight=weight.to('cpu'))*ibatch
    val_loss /= nProcessed
    val_acc  /= nProcessed

    if bestLoss > val_loss:
        bestState = model.to('cpu').state_dict()
        bestLoss = val_loss
        torch.save(bestState, os.path.join('result/' + args.output, 'weight.pth'))

        model.to(device)

    train['loss'].append(trn_loss)
    train['acc'].append(trn_acc)
    train['val_loss'].append(val_loss)
    train['val_acc'].append(val_acc)

    with open(os.path.join('result/' + args.output, 'train.csv'), 'w') as f:
        writer = csv.writer(f)
        keys = train.keys()
        writer.writerow(keys)
        for row in zip(*[train[key] for key in keys]):
            writer.writerow(row)

bestState = model.to('cpu').state_dict()
torch.save(bestState, os.path.join('result/' + args.output, 'weightFinal.pth'))


