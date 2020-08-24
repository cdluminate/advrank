'''
Copyright (C) 2019-2020, Authors of ECCV2020 #2274 "Adversarial Ranking Attack and Defense"
Copyright (C) 2019-2020, Mo Zhou <cdluminate@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import os
import torch as th
import collections
import math
import yaml
from . import datasets
from . import faC_c2f2_siamese
import numpy as np

class Model(faC_c2f2_siamese.Model):
    """
    LeNet-like convolutional neural network for embedding
    """
    def loss(self, x, y, *, marginC=0.2, marginE=1.0, hard=True):
        '''
        Input[x]: images [N, 1, 28, 28]
        Input[y]: corresponding image labels [N]
        '''
        x = x.to(self.fc1.weight.device)
        y = y.to(self.fc1.weight.device).view(-1)
        if self.metric == 'C':
            output = self.forward(x, l2norm=True)
        elif self.metric == 'E':
            output = self.forward(x, l2norm=False)

        # -- alternative version: hard contrastive sampling, or random sampling
        # idxp = same class, largest distance (as the hard positive example)
        # idxn = diff class, smallest distance (as the hard negative example)
        # loss = triplet(output, output[idxp], output[idxn], ...)

        if self.metric == 'C':
            Y = y.cpu().numpy()
            idxa, idxp, idxn = [], [], []
            for i in range(10): # 10 classes
                ida = np.random.choice(np.where(Y == i)[0],
                        int(output.shape[0]/10), replace=True)
                idp = np.random.choice(np.where(Y == i)[0],
                        int(output.shape[0]/10), replace=True)
                idn = np.random.choice(np.where(Y != i)[0],
                        int(output.shape[0]/10), replace=True)
                idxa.extend(list(ida))
                idxp.extend(list(idp))
                idxn.extend(list(idn))
            XA = output[th.LongTensor(idxa)]
            XP = output[th.LongTensor(idxp)]
            XN = output[th.LongTensor(idxn)]
            DAP = 1 - th.nn.functional.cosine_similarity(XA, XP)
            DAN = 1 - th.nn.functional.cosine_similarity(XA, XN)
            loss = (DAP - DAN + marginC).clamp(min=0.).mean()
        elif self.metric == 'E':
            Y = y.cpu().numpy()
            idxa, idxp, idxn = [], [], []
            for i in range(10): # 10 classes
                ida = np.random.choice(np.where(Y == i)[0],
                        int(output.shape[0]/10), replace=True)
                idp = np.random.choice(np.where(Y == i)[0],
                        int(output.shape[0]/10), replace=True)
                idn = np.random.choice(np.where(Y != i)[0],
                        int(output.shape[0]/10), replace=True)
                idxa.extend(list(ida))
                idxp.extend(list(idp))
                idxn.extend(list(idn))
            XA = output[th.LongTensor(idxa)]
            XP = output[th.LongTensor(idxp)]
            XN = output[th.LongTensor(idxn)]
            loss = th.nn.functional.triplet_margin_loss(XA, XP, XN,
                    p=2, margin=marginE, reduction='mean')
        else:
            raise ValueError(self.metric)

        return output, loss
