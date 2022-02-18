#!/usr/bin/env python
import torch.nn as nn
import numpy as np
import torch

# device = "cuda:2"
class WF1DCNN3FC2logModel(nn.Module):
    def __init__(self, **kwargs):
        super(WF1DCNN3FC2logModel, self).__init__()

        self.nPt = kwargs['nPoint']
        self.nCh = 1 if 'nChannel' not in kwargs else kwargs['nChannel']
        nPt = self.nPt
        kernel1 = 11 if 'kernel_size' not in kwargs else kwargs['kernel_size']
        self.device = "cuda:" + str(kwargs['device'])
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.nCh, 64, kernel_size=kernel1),
            nn.MaxPool1d(kernel1, stride=kernel1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
        )
        nPt = (nPt-kernel1+1)//kernel1
        #print(nPt)

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),
            nn.MaxPool1d(2, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
        )
        nPt = (nPt-3+1)//2

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3),
            nn.MaxPool1d(2, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
        )
        nPt = (nPt-3+1)//2
        
        self.fc = nn.Sequential(
            nn.Linear(nPt*256, 512),
            nn.ReLU(), nn.Dropout(0.5),

            nn.Linear(512, 512),
            nn.ReLU(), nn.Dropout(0.5),

            nn.Linear(512, 1),
        )

    def forward(self, x):
        batch, n = x.shape[0], x.shape[1]
        #assert (n == self.nPt)
        #assert (c == self.nCh)
#         x = x/x.max()

#         y = torch.abs(x)

#         x = -torch.sign(x)*torch.log(y)
       
#         x = x.to(self.device)
#         x = x.to(device)
#         x = x/x.max()

        y = np.abs(x.to("cpu"))

        x = -np.sign(x.to("cpu"))*np.ma.log(y.to("cpu")).filled(0)
        x = x.to(self.device)
#         t = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        

        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x

