#!/usr/bin/env python
import torch.nn as nn

class WF1DCNN3FC2Model(nn.Module):
    def __init__(self, **kwargs):
        super(WF1DCNN3FC2Model, self).__init__()

        self.nPt = kwargs['nPoint']
        self.nCh = 1 if 'nChannel' not in kwargs else kwargs['nChannel']
        #self.nCh = 1 if 'channel' not in kwargs else kwargs['channel']
        nPt = self.nPt
        kernel1 = 8 if 'kernel_size' not in kwargs else kwargs['kernel_size']

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
        x = x/x.max()
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        

        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x

