#!/usr/bin/env pythnon

#### normalize data by each wave maximum value(one of 96 waveform)

import h5py
import torch
from torch.utils.data import Dataset
from bisect import bisect_right
from glob import glob
import pandas as pd
import numpy as np

class WFCh2Dataset_each_norm_involve_raw(Dataset):
    def __init__(self, **kwargs):
        super(WFCh2Dataset_each_norm_involve_raw, self).__init__()
        self.isLoaded = False
        self.fNames = []
        self.sampleInfo = pd.DataFrame(columns=["procName", "fileName", "weight", "label", "fileIdx"])
        
        self.channel = kwargs['channel'] if 'channel' in kwargs else 1
        self.output = kwargs['output']
        #self.width = int(496/0.2)
        self.dataName = 'waveform'
        self.width = 208
#         self.dataName = 'fadcs'

    def __getitem__(self, idx):
        if not self.isLoaded: self.load()

        fileIdx = bisect_right(self.maxEventsList, idx)-1
        offset = self.maxEventsList[fileIdx]
        idx = idx-int(offset)
#         print('fileIdx:', fileIdx, '/' , 'idx:', idx)
        
        image  = self.imagesList[fileIdx][idx]
        label  = self.labelsList[fileIdx][idx]
        weight = self.weightsList[fileIdx][idx]
        
        dVertexx = self.dVertexList[fileIdx][idx]
        minvaluee = self.minvalueList[fileIdx][idx]
        pChargee = self.pChargeList[fileIdx][idx]
     
        rescale = self.rescaleList[fileIdx][idx]
        procIdx = self.procList[fileIdx][idx]
        
        vertexx = self.vertexXList[fileIdx][idx]
        vertexy = self.vertexYList[fileIdx][idx]
        vertexz = self.vertexZList[fileIdx][idx]
        
        image = np.where(image < -1000, 0, image)
        image /= image.max()


        return (image, label, weight, rescale, procIdx, fileIdx, idx, dVertexx, minvaluee, pChargee,vertexx,vertexy,vertexz)

    def __len__(self):
        return int(self.maxEventsList[-1])

    def addSample(self, procName, fNamePattern, weight=None, logger=None):
        if logger: logger.update(annotation='Add sample %s <= %s' % (procName, fileNamePattern))
        weightValue = weight ## Rename it just to be clear in the codes

        for fName in glob(fNamePattern):
            if not fName.endswith(".h5"): continue
            fileIdx = len(self.fNames)
            self.fNames.append(fName)
  
            info = {
                'procName':procName, 'weight':weight, 'nEvents':0,
                'label':0, ## default label, to be filled later
                'fileName':fName, 'fileIdx':fileIdx,
            }
            self.sampleInfo = self.sampleInfo.append(info, ignore_index=True)

    def setProcessLabel(self, procName, label):
        self.sampleInfo.loc[self.sampleInfo.procName==procName, 'label'] = label

    def initialize(self, logger=None):
        if logger: logger.update(annotation='Reweights by category imbalance')
        procNames = list(self.sampleInfo['procName'].unique())

        self.labelsList = []
        self.weightsList = []
        self.rescaleList = []
        self.procList = []
        self.imagesList = []
        
        self.dVertexList = []
        self.minvalueList = []
        self.pChargeList = []
        
        self.vertexXList = []
        self.vertexYList = []
        self.vertexZList = []

        nFiles = len(self.sampleInfo)
        print(self.sampleInfo, 'SI')
        ## Load event contents
        for i, fName in enumerate(self.sampleInfo['fileName']):
    
            data = h5py.File(fName, 'r', libver='latest', swmr=True)['events']

            images  = data[self.dataName]
            nEvents = images.shape[0]
            self.sampleInfo.loc[i, 'nEvents'] = nEvents
     
            weightValue = self.sampleInfo.loc[i, 'weight']


            dVertex = data['dVertex']
            pCharge = data['pCharge']
            minvalue = data['minvalue']
            
            vtxx = data['vertexX']
            vtxy = data['vertexY']
            vtxz = data['vertexZ']
            
            if weightValue is None: weights = data['weights']
            else: weights = torch.ones(nEvents, dtype=torch.float32, requires_grad=False)*weightValue

            ## set label and weight
            label = self.sampleInfo['label'][i]
            labels = torch.ones(nEvents, dtype=torch.int32, requires_grad=False)*label
            
          
            
            self.labelsList.append(labels)
            weight = self.sampleInfo['weight'][i]
   

            
            #####
            weights = torch.ones(nEvents, dtype=torch.float32, requires_grad=False)*weight

            self.weightsList.append(weights)
            self.rescaleList.append(torch.ones(nEvents, dtype=torch.float32, requires_grad=False))
            procIdx = procNames.index(self.sampleInfo['procName'][i])
            self.procList.append(torch.ones(nEvents, dtype=torch.int32, requires_grad=False)*procIdx)
            
            self.dVertexList.append(dVertex)
            self.pChargeList.append(pCharge)
            self.minvalueList.append(minvalue)
            
            self.vertexXList.append(vtxx)
            self.vertexYList.append(vtxy)
            self.vertexZList.append(vtxz)

        SI = self.sampleInfo
        #### save sampleInfo file in train result path
        SI.to_csv(self.output + '/sampleInfo.csv')
      
        ## Compute cumulative sums of nEvents, to be used for the file indexing
        self.maxEventsList = np.concatenate(([0.], np.cumsum(self.sampleInfo['nEvents'])))

        ## Compute sum of weights for each label categories
        sumWByLabel = {}
        sumEByLabel = {}
        for label in self.sampleInfo['label']:
            label = int(label)
            w = self.sampleInfo[self.sampleInfo.label==label]['weight']
            
            e = self.sampleInfo[self.sampleInfo.label==label]['nEvents']
#             print(e, 'e')
            sumWByLabel[label] = (w*e).sum()
            sumEByLabel[label] = e.sum()
        ## Find overall rescale for the data imbalancing problem - fit to the category with maximum entries
        maxSumELabel = max(sumEByLabel, key=lambda key: sumEByLabel[key])
        maxWMaxSumELabel = self.sampleInfo[self.sampleInfo.label==maxSumELabel]['weight'].max()
        minWMaxSumELabel = self.sampleInfo[self.sampleInfo.label==maxSumELabel]['weight'].min()
        avgWgtMaxSumELabel = sumWByLabel[maxSumELabel]/sumEByLabel[maxSumELabel]

        ## Find overall rescale for the data imbalancing problem - fit to the category with maximum entries
        #### Find rescale factors - make weight to be 1 for each cat in the training step
        for fileIdx in self.sampleInfo['fileIdx']:
            label = self.sampleInfo.loc[self.sampleInfo.fileIdx==fileIdx, 'label']
            for l in label: ## this loop runs only once, by construction.
                self.rescaleList[fileIdx] *= (sumEByLabel[maxSumELabel]/sumWByLabel[l])
                #print("@@@ Scale sample label_%d(sumE=%g,sumW=%g)->label_%d, sf=%f" % (l, sumEByLabel[l], sumWByLabel[l], maxSumELabel, sf))
                break ## this loop runs only once, by construction. this break is just for a confirmation
        
        print('-'*80)
        for label in sumWByLabel.keys():
            print("Label=%d sumE=%d, sumW=%g" % (label, sumEByLabel[label], sumWByLabel[label]))
        print('Label with maxSumE:%d' % maxSumELabel)
        print('      maxWeight=%g minWeight=%g avgWeight=%g' % (maxWMaxSumELabel, minWMaxSumELabel, avgWgtMaxSumELabel))
        print('-'*80)
    
    def load(self):
        if self.isLoaded: return
        for fName in list(self.sampleInfo['fileName']):
            image = h5py.File(fName, 'r', libver='latest', swmr=True)['events/'+self.dataName]
            #print(image.shape)
#             image = image[()].reshape(-1, self.channel, self.width)
            image = image[()]
            image = image[:,:,-209:-1]
       
            self.imagesList.append(image)
        self.isLoaded = True

