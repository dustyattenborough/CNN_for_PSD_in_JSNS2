#!/usr/bin/env python
import sys, os
import argparse

import numpy as np
import pandas as pd
import uproot
import csv, yaml

import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--config', action='store', type=str, default='config.yaml', help='Configration file with sample information')
parser.add_argument('--data', action='store', type=str, help='data type select')
parser.add_argument('-o', '--output', action='store', type=str, required=True, help='Path to output file')
parser.add_argument('--mdinput', action='store', type=str, required=True, help='Path model weight')

parser.add_argument('--odd', action='store', type=float, help='odd =1 or total = 0')
parser.add_argument('--runnum', action='store', type=int, help='runnumber')
parser.add_argument('--rho', action='store', type=float, help='rho')
parser.add_argument('--vtz', action='store', type=float, help='vtz')

parser.add_argument('--device', action='store', type=int, default=0, help='device name')
parser.add_argument('--batch', action='store', type=int, default=64, help='Batch size')

args = parser.parse_args()
if not os.path.exists('result/'+args.output): os.makedirs('result/'+args.output)
config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)


sys.path.append("./python")

torch.set_num_threads(os.cpu_count())
if torch.cuda.is_available() and args.device >= 0: torch.cuda.set_device(args.device)

##### Define dataset instance #####

from dataset.WFCh2Dataset_each_norm import *
from dataset.WFCh2Dataset_max_norm import *
from dataset.WFCh2Dataset_each_norm_involve_raw import *

##### Define dataset instance #####
dset = WFCh2Dataset_each_norm_involve_raw(channel=config['format']['channel'], output = 'result/' + args.output)
for sampleInfo in config['samples']:
    if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
    name = sampleInfo['name']
    if args.odd == 1:
        in_path = '/users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00'+str(args.runnum) + '_v3_ch/'+name+'_even_Rho_1.4_ZL_1.0_min_400_dv_1/*.h5'
    elif args.odd == 2:
        in_path = '/users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00'+str(args.runnum) + '_v3_ch/'+name+'_odd_Rho_1.4_ZL_1.0_min_400_dv_1/*.h5'
    
    else: ## total training
        in_path = '/users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00'+str(args.runnum) + '_v3_ch/'+name+'_cut_Rho_1.4_ZL_1.0_min_400_dv_1/*.h5'

        
        
    dset.addSample(name, in_path, weight=sampleInfo['xsec']/sampleInfo['ngen'])
#     dset.addSample(name, sampleInfo['path'], weight=sampleInfo['xsec']/sampleInfo['ngen'])
    dset.setProcessLabel(name, sampleInfo['label'])
dset.initialize()
lengths = [int(x*len(dset)) for x in config['training']['splitFractions']]
lengths.append(len(dset)-sum(lengths))

kwargs = {'num_workers':min(config['training']['nDataLoaders'], os.cpu_count()),
          'batch_size':args.batch, 'pin_memory':False}
from torch.utils.data import DataLoader
trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)
#testLoader = DataLoader(trnDset, **kwargs)
#testLoader = DataLoader(valDset, **kwargs)
testLoader = DataLoader(testDset, **kwargs)

##### Define model instance #####
from models.allModels import *
#model = WF1DCNNModel(nChannel=dset.nCh, nPoint=dset.nPt)
model = torch.load( args.mdinput+'/model.pth', map_location='cpu')
model.load_state_dict(torch.load(args.mdinput+'/weight.pth', map_location='cpu'))

model.fc.add_module('output', torch.nn.Sigmoid())

device = 'cpu'
if args.device >= 0 and torch.cuda.is_available():
    model = model.cuda()
    device = 'cuda'



##### Start evaluation #####
from tqdm import tqdm
labels, preds = [], []
weights = []
procIdxs = []
fileIdxs = []
idxs = []

minvalues = []
dVertexsss = []
pCharges = []

vertexXs = []
vertexYs = []
vertexZs = []

model.eval()
val_loss, val_acc = 0., 0.
# for i, (data, label0, weight, rescale, procIdx, fileIdx, idx, dT, dVertex, vertexX, vertexY, vertexZ) in enumerate(tqdm(testLoader)):

for i, (data, label0, weight, rescale, procIdx, fileIdx, idx, dVertexx, minvaluee, pChargee,vertexx,vertexy,vertexz) in enumerate(tqdm(testLoader)):
    data = data.to(device)


    label = label0.to(device)
    label0 = label0.reshape(-1)[()].numpy()
    rescale = rescale.float().to(device)
    weight = weight.float().to(device)*rescale

    pred = model(data)

    dVertex = dVertexx.to(device)
    minvalue = minvaluee.to(device)
    pCharge = pChargee.to(device)
    
    vertexX = vertexx.to(device)
    vertexY = vertexy.to(device)
    vertexZ = vertexz.to(device)

    vertexXs.extend([x.item() for x in vertexX])
    vertexYs.extend([x.item() for x in vertexY])
    vertexZs.extend([x.item() for x in vertexZ])
  
    pCharges.extend([x.item() for x in pCharge.view(-1)])
    minvalues.extend([x.item() for x in minvalue])
    dVertexsss.extend([x.item() for x in dVertex])
    
    labels.extend([x.item() for x in label])
    weights.extend([x.item() for x in weight])
    preds.extend([x.item() for x in pred.view(-1)])
    procIdxs.extend([x.item() for x in procIdx])
    fileIdxs.extend([x.item() for x in fileIdx])
    idxs.extend([x.item() for x in idx])

df = pd.DataFrame({'label':labels, 'prediction':preds, 'weight':weights, 'procIdx':procIdxs, 'fileIdx':fileIdxs, 'idx':idxs})

fPred = 'result/' + args.output + '/' + args.output + '.csv'
df.to_csv(fPred, index=False)



##### Draw ROC curve #####
si_path = 'result/' + args.output + '/sampleInfo.csv'
si = pd.read_csv(si_path)
fPred = 'result/' + args.output + '/' + args.output + '.csv'
info = pd.read_csv(fPred)

info_numpy = np.array(info)
si_numpy = np.array(si)

preds = []
labels = []
dVertexs = []
dTs = []
vertexXs = []
vertexYs = []
vertexZs = []
for i in range(len(info_numpy)):

    label = info_numpy[i][0]
    pred = info_numpy[i][1]

    fileidx = info_numpy[i][4]
    
    filename = si_numpy[int(fileidx)][2]
    
    
    idx = info_numpy[i][5]

    
    data = h5py.File(filename,'r')

    dT = data['events']['dT'][idx]
    dVertex = data['events']['dVertex'][idx]
    vertexX = data['events']['vertexX'][idx]
    vertexY = data['events']['vertexY'][idx]
    vertexZ = data['events']['vertexZ'][idx]
    
    
    preds.append(pred)
    labels.append(label)
    dVertexs.append(dVertex)
    dTs.append(dT)
#     print(dTs.type)
    vertexXs.append(vertexX)
    vertexYs.append(vertexY)
    vertexZs.append(vertexZ)

preds = np.array(preds)
labels = np.array(labels)
dVertexs = np.array(dVertexs)
dTs = np.array(dTs)
vertexXs = np.array(vertexXs)
vertexYs = np.array(vertexYs)
vertexZs = np.array(vertexZs)


ME_label = []
ME_dVertex = []
ME_dT = []
ME_vertexX = []
ME_vertexY = []
ME_vertexZ = []
ME_pred = []


FN_label = []
FN_dVertex = []
FN_dT = []
FN_vertexX = []
FN_vertexY = []
FN_vertexZ = []
FN_pred = []

for i in range(len(preds)):
    if labels[i] == 1:
        ME_label.append(labels[i])
        ME_dVertex.append(dVertexs[i])
        ME_dT.append(dTs[i])
        ME_vertexX.append(vertexXs[i])
        ME_vertexY.append(vertexYs[i])
        ME_vertexZ.append(vertexZs[i])
        ME_pred.append(preds[i])
    else:
   
        FN_label.append(labels[i])
        FN_dVertex.append(dVertexs[i])
        FN_dT.append(dTs[i])
        FN_vertexX.append(vertexXs[i])
        FN_vertexY.append(vertexYs[i])
        FN_vertexZ.append(vertexZs[i])
        FN_pred.append(preds[i])
        
ME_sig = -np.log((1/np.array(ME_pred))-1)
FN_sig = -np.log((1/np.array(FN_pred))-1)
FN_over_0 = 100*np.sum(FN_sig > 0)/len(FN_sig)
ME_under_0 = 100*np.sum(ME_sig < 0)/len(ME_sig)


##################plot CNN score distribution figure
plt.hist(FN_sig, bins = 100, range = (-20, 20), density = True, color ='r',histtype = 'step')
plt.hist(ME_sig, bins = 100, range = (-20, 20), density = True, color ='b',histtype = 'step')
# plt.text(10, 0.15,'FN > 0' + str(FN_over_0), fontsize = 20)
# plt.text(10, 0.1, 'ME < 0' + str(ME_under_0), fontsize = 20)


FN_over = '%.2f' %FN_over_0
ME_under = '%.2f' %ME_under_0

label = ['FN : FN > 0 = '+FN_over+'%', 
         'ME : ME < 0 = '+ME_under+'%']


leg = plt.legend(label, loc = 'best', frameon=False)

leg_lines = leg.get_lines()
leg_texts = leg.get_texts()

plt.setp(leg_lines, linewidth=15)
plt.setp(leg_texts, fontsize=15)

plt.title('')
plt.savefig('result/' + args.output + '/' + args.output + '_CNN score distribution.png', dpi=300)
plt.clf()



num_FN = len(FN_dT)
num_ME = len(ME_dT)
###########plot dT
plt.hist(np.array(FN_dT)*0.001, bins = 100, color= 'r', alpha = 0.5, density = True, histtype = 'step')
plt.hist(np.array(ME_dT)*0.001, bins = 100, color= 'b', alpha = 0.5, density = True, histtype = 'step')

plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel("\u03BCs", fontsize=15, loc='right')
plt.savefig('result/' + args.output + '/' + args.output + '_dT.png', dpi=300)
plt.clf()

###########plot dVertex
plt.hist(np.array(FN_dVertex)*100, bins = 80, color= 'r', alpha = 0.5, density = True, histtype = 'step')
plt.hist(np.array(ME_dVertex)*100, bins = 80, color= 'b', alpha = 0.5, density = True, histtype = 'step')

plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel("cm", fontsize=15, loc='right')
plt.savefig('result/' + args.output + '/' + args.output + '_dVertex.png', dpi=300)
plt.clf()


##############plot vertexZ
plt.hist(np.array(FN_vertexZ)*100, bins = 80, color= 'r', alpha = 0.5, density = True, histtype = 'step')
plt.hist(np.array(ME_vertexZ)*100, bins = 80, color= 'b', alpha = 0.5, density = True, histtype = 'step')

plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel("cm", fontsize=15, loc='right')
plt.savefig('result/' + args.output + '/' + args.output + '_vertexZ.png', dpi=300)
plt.clf()


##############plot R2
plt.hist((np.array(FN_vertexX)**2+np.array(FN_vertexY)**2)*100, bins = 80, color= 'r', alpha = 0.5, density = True, histtype = 'step')
plt.hist((np.array(ME_vertexX)**2+np.array(ME_vertexY)**2)*100, bins = 80, color= 'b', alpha = 0.5, density = True, histtype = 'step')

plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel("cm", fontsize=15, loc='right')
plt.savefig('result/' + args.output + '/' + args.output + '_vertex_R2.png', dpi=300)
plt.clf()

####################################################################################

list_range = np.arange(0,1,0.001)


for i in range(len(list_range)):
    a = len(np.array(ME_pred)[np.array(ME_pred)>list_range[i]])/len(np.array(ME_pred))
#     print(list_range[i])
    
    if (a > 0.99):
     
        eff_99 = list_range[i]
        continue
        
    if (a > 0.95):
        
        eff_95 = list_range[i]
        continue
        
    if (a > 0.90):
       
        eff_90 = list_range[i]
        continue
##### eff 0.5 pred
eff_50_ME = len(np.array(ME_pred)[np.array(ME_pred)>0.5])/len(np.array(ME_pred))
eff_50_FN = len(np.array(FN_pred)[np.array(FN_pred)>0.5])/len(np.array(FN_pred))
num_50_ME = num_ME*len(np.array(ME_pred)[np.array(ME_pred)>0.5])/len(np.array(ME_pred))
num_50_FN = num_FN*len(np.array(FN_pred)[np.array(FN_pred)>0.5])/len(np.array(FN_pred))



##### eff 99% efficient
eff_99_ME = len(np.array(ME_pred)[np.array(ME_pred)>eff_99])/len(np.array(ME_pred))
eff_99_FN = len(np.array(FN_pred)[np.array(FN_pred)>eff_99])/len(np.array(FN_pred))
num_99_ME = num_ME*len(np.array(ME_pred)[np.array(ME_pred)>eff_99])/len(np.array(ME_pred))
num_99_FN = num_FN*len(np.array(FN_pred)[np.array(FN_pred)>eff_99])/len(np.array(FN_pred))




##### eff 95% efficient
eff_95_ME = len(np.array(ME_pred)[np.array(ME_pred)>eff_95])/len(np.array(ME_pred))
eff_95_FN = len(np.array(FN_pred)[np.array(FN_pred)>eff_95])/len(np.array(FN_pred))
num_95_ME = num_ME*len(np.array(ME_pred)[np.array(ME_pred)>eff_95])/len(np.array(ME_pred))
num_95_FN = num_FN*len(np.array(FN_pred)[np.array(FN_pred)>eff_95])/len(np.array(FN_pred))


##### eff 90% efficient
eff_90_ME = len(np.array(ME_pred)[np.array(ME_pred)>eff_90])/len(np.array(ME_pred))
eff_90_FN = len(np.array(FN_pred)[np.array(FN_pred)>eff_90])/len(np.array(FN_pred))
num_90_ME = num_ME*len(np.array(ME_pred)[np.array(ME_pred)>eff_90])/len(np.array(ME_pred))
num_90_FN = num_FN*len(np.array(FN_pred)[np.array(FN_pred)>eff_90])/len(np.array(FN_pred))


f = open('result/' + args.output + '/' + args.output + '_efficiency.txt','w')
print('         |   90%   |   95%   |   99%   |   mid   |',file=f)
print('--------------------------------------------------',file=f)
print('  ME_eff |','%.4f  |'%eff_90_ME,'%.4f  |'%eff_95_ME,'%.4f  |'%eff_99_ME,'%.4f  |'%eff_50_ME,file=f)
print('--------------------------------------------------',file=f)
print('  FN_eff |','%.4f  |'%eff_90_FN,'%.4f  |'%eff_95_FN,'%.4f  |'%eff_99_FN,'%.4f  |'%eff_50_FN,file=f)
print('--------------------------------------------------',file=f)
print('ME_remain|','%7d'%int(num_90_ME),'|','%7d'%int(num_95_ME),'|','%7d'%int(num_99_ME),'|','%7d'%int(num_50_ME),'|',file=f)
print('--------------------------------------------------',file=f)
print('FN_remain|','%7d'%int(num_90_FN),'|','%7d'%int(num_95_FN),'|','%7d'%int(num_99_FN),'|','%7d'%int(num_50_FN),'|',file=f)
print('--------------------------------------------------',file=f)
print('CNN score|','','%.3f'%eff_90,' | ','%.3f'%eff_95,' | ','%.3f'%eff_99,' |  ',0.5,'  |',file=f)
print('==================================================',file=f)
print('         |   FN    |   ME    |',file=f)
print('# data   |','%7d'%len(FN_dT),'|', '%7d'%len(ME_dT),'|',file=f)

f.close()



###############plot evaluation 

##### Draw ROC curve #####
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib.cbook import get_sample_data

df = pd.read_csv(fPred)
tpr, fpr, thr = roc_curve(df['label'], df['prediction'], sample_weight=df['weight'], pos_label=0)
auc = roc_auc_score(df['label'], df['prediction'], sample_weight=df['weight'])


df_bkg = df[df.label==0]
df_sig = df[df.label==1]
plt.rcParams['figure.figsize'] = (10, 10)

plt.hist(df_bkg['prediction']*100, weights=df_bkg['weight'], histtype='step', 
         density=True, bins=50, color='red', linewidth=3)

plt.hist(df_sig['prediction']*100, weights=df_sig['weight'], histtype='step', 
         density=True, bins=50, color='blue', linewidth=3)



plt.xticks(np.arange(0, 101, step=20),["{}".format(x*0.01) for x in np.arange(0, 101,step=20)],fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel("CNN score", fontsize=15, loc='right')
plt.ylabel("Normalized", fontsize=15, loc='top')
plt.xlim(0, 100)
label = ['Fast Neutron', 'Michel Electrons']

leg = plt.legend(label, loc = 'upper center', frameon=False)

leg_lines = leg.get_lines()
leg_texts = leg.get_texts()

plt.setp(leg_lines, linewidth=10)
plt.setp(leg_texts, fontsize=10)

plt.savefig('result/' + args.output + '/' + args.output + '_evaluation.png', dpi=300)
plt.clf()



###################### plot AUC


plt.rcParams['figure.figsize'] = (15, 15)



plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)



plt.plot(fpr*100, tpr*100, label='AUC = %.3f' % (auc))
plt.plot(eff_99_FN*100, eff_99_ME*100,'*r', markersize=40)

plt.plot(eff_95_FN*100, eff_95_ME*100,'*g', markersize=40)

plt.plot(eff_90_FN*100, eff_90_ME*100,'*b', markersize=40)

plt.plot(eff_50_FN*100, eff_50_ME*100,'*k', markersize=40)

plt.xlabel('FN efficiency (%)', fontsize=15, loc='right')
plt.ylabel('ME efficiency (%)', fontsize=15, loc='top')


plt.xlim(0, 100)
plt.ylim(0, 100)

plt.grid()
print_auc = '%.3f' %auc
print_eff_99 = '%.3f' %eff_99
print_eff_95 = '%.3f' %eff_95
print_eff_90 = '%.3f' %eff_90
label = ['AUC = '+print_auc,'99% WP(score='+print_eff_99+')', '95% WP(score='+print_eff_95+')', '90% WP(score='+print_eff_90+')','Middle WP (score=0.5)']

leg = plt.legend(label, loc = 'right', frameon=False)

 
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()

plt.setp(leg_lines, linewidth=10)
plt.setp(leg_texts, fontsize=10)

plt.savefig('result/' + args.output + '/' + args.output + '_AUC.png', dpi=300)
plt.clf()


f = open('result/' + args.output + '/' + args.output + '_information.txt','w')


print(eff_90_ME,eff_95_ME,eff_99_ME,eff_50_ME,file=f)

print(eff_90_FN,eff_95_FN,eff_99_FN,eff_50_FN,file=f)

print(int(num_90_ME),int(num_95_ME),int(num_99_ME),int(num_50_ME),file=f)

print(int(num_90_FN),int(num_95_FN),int(num_99_FN),int(num_50_FN),file=f)

print(eff_90,eff_95,eff_99,0.5,file=f)


print(len(FN_dT),len(ME_dT),file=f)
print(auc,file=f)

f.close()


# #########plot dT vs CNN score   FN
# from scipy.stats import gaussian_kde
# xy = np.vstack([np.array(FN_dT)*0.001,np.array(FN_sig)])
# xy[xy==np.inf] =0
# z = gaussian_kde(xy)(xy)
# idx = z.argsort()
# x, y, z = (np.array(FN_dT)*0.001)[idx], np.array(FN_sig)[idx], z[idx]


# plt.scatter(x, y, c=z, s=50, cmap=plt.cm.jet)

# plt.xticks(fontsize = 15)
# plt.yticks(fontsize = 15)
# plt.xlabel("\u03BCs", fontsize=25, loc='right')
# plt.ylabel("score", fontsize=25, loc='top')
# plt.colorbar()
# plt.savefig('result/' + args.output + '/' + args.output + '_FN_score_scatter.png', dpi=300)
# plt.clf()


# #########plot dT vs CNN score    ME
# xy = np.vstack([np.array(ME_dT)*0.001,np.array(ME_sig)])
# xy[xy==np.inf] =0
# z = gaussian_kde(xy)(xy)
# idx = z.argsort()
# x, y, z = (np.array(ME_dT)*0.001)[idx], np.array(ME_sig)[idx], z[idx]



# plt.scatter(x, y, c=z, s=50, cmap=plt.cm.jet)

# plt.xticks(fontsize = 15)
# plt.yticks(fontsize = 15)
# plt.xlabel("\u03BCs", fontsize=25, loc='right')
# plt.ylabel("score", fontsize=25, loc='top')
# # plt.ylim([-50, 50])
# plt.colorbar()
# plt.savefig('result/' + args.output + '/' + args.output + '_ME_score_scatter.png', dpi=300)
# plt.clf()


# ############ FN mean plot
# FN_sig = np.array(FN_sig)
# FN_sig_test = FN_sig[FN_sig>-100]
# FN_dT_test = np.array(FN_dT)*0.001
# FN_dT_test = FN_dT_test[FN_sig>-100]
# ##### seperate range
# mm = []
# rr = 10
# for i in range(100, 10000, rr):
# #     print(i/100)
#     a = np.mean(FN_sig_test[(FN_dT_test< (i/100 + 0.10)) & (FN_dT_test>(i/100))])
#     mm.append(a)
# tt = np.array(list(range(100, 10000, rr)))/100
# plt.scatter(tt, mm)
# plt.ylim([-15,5])
# plt.xticks(fontsize = 15)
# plt.yticks(fontsize = 15)
# plt.xlabel("\u03BCs", fontsize=25, loc='right')
# plt.ylabel("score", fontsize=25, loc='top')
# plt.savefig('result/' + args.output + '/' + args.output + '_FN_mean.png', dpi=300)
# plt.clf()

# ############ ME mean plot
# ME_sig = np.array(ME_sig)
# ME_sig_test = ME_sig[ME_sig<100]
# ME_dT_test = np.array(ME_dT)*0.001
# ME_dT_test = ME_dT_test[ME_sig<100]
# ##### seperate range
# mm = []
# rr = 10
# for i in range(100, 1000, rr):
# #     print(i/100)
#     a = np.mean(ME_sig_test[(ME_dT_test< (i/100 + 0.10)) & (ME_dT_test>(i/100))])
#     mm.append(a)
# tt = np.array(list(range(100, 1000, rr)))/100
# plt.scatter(tt, mm)
# plt.ylim([-5,15])
# plt.xticks(fontsize = 15)
# plt.yticks(fontsize = 15)
# plt.xlabel("\u03BCs", fontsize=25, loc='right')
# plt.ylabel("score", fontsize=25, loc='top')
# plt.savefig('result/' + args.output + '/' + args.output + '_ME_mean.png', dpi=300)
# plt.clf()


# ############ FN 2dhist plot
# plt.hist2d(np.array(FN_dT)[FN_sig>-100]*0.001,np.array(FN_sig)[FN_sig>-100],(50,50))
# plt.ylim([-20, 20])
# plt.savefig('result/' + args.output + '/' + args.output + '_FN_2dhist.png', dpi=300)
# plt.clf()


# ############ ME 2dhist plot
# plt.hist2d(np.array(ME_dT)[ME_sig<100]*0.001,np.array(ME_sig)[ME_sig<100],(50,50))
# plt.ylim([-20, 20])
# plt.savefig('result/' + args.output + '/' + args.output + '_ME_2dhist.png', dpi=300)
# plt.clf()



