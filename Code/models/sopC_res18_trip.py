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
from . import datasets, common
import torchvision as vision
from tqdm import tqdm
import numpy as np
from . import sopC_res18_siamese

class Model(sopC_res18_siamese.Model):
    """
    Res18 + SOP dataset, Triplet Loss function
    """
    def loss(self, x, y, *, marginC=0.5, marginE=1.0):
        '''
        Input[x]: image triplets in shape [3N, 3, 224, 224]
        Input[y]: not used
        '''
        x = x.to(self.device).view(-1, 3, 224, 224)
        #y = y.to(self.device).view(-1) # y is not used here
        output = self.forward(x)
        if self.metric == 'C': # Cosine
            pdistAP = 1 - th.nn.functional.cosine_similarity(
                    output[0::3], output[1::3], dim=1)
            pdistAN = 1 - th.nn.functional.cosine_similarity(
                    output[0::3], output[2::3], dim=1)
            # compute loss: triplet margin loss, cosine version
            loss = (pdistAP - pdistAN + marginC).clamp(min=0.).sum()
        else: # Euclidean
            loss = th.nn.functional.triplet_margin_loss(
                    output[0::3], output[1::3], output[2::3],
                    margin=marginE, p=2, reduction='sum')
        return output, loss

    def report(self, epoch, iteration, total, output, labels, loss):
        if self.metric == 'C': # Cosine
            pdistAP = 1-th.nn.functional.cosine_similarity(output[0::3],output[1::3],dim=1)
            pdistAN = 1-th.nn.functional.cosine_similarity(output[0::3],output[2::3],dim=1)
        else: # Euclidean
            pdistAP = th.nn.functional.pairwise_distance(output[0::3],output[1::3])
            pdistAN = th.nn.functional.pairwise_distance(output[0::3],output[2::3])
        ineqacc = (pdistAP < pdistAN).float().mean()
        print(f'Eph[{epoch}][{iteration}/{total}]',
                collections.namedtuple('Res', ['loss', 'ineqacc'])(
                    '%.4f'%loss.item(), '%.4f'%ineqacc.item()))

    def getloader(self, kind:str='train', batchsize:int=1):
        '''
        get corresponding dataloaders
        '''
        config = yaml.load(open('config.yml', 'r').read(),
                Loader=yaml.SafeLoader)
        if kind == 'train':
            return datasets.sop.get_loader(
                    os.path.expanduser(config['sop']['path']),
                    batchsize, 3, 'train')
        else:
            return datasets.sop.get_loader(
                    os.path.expanduser(config['sop']['path']),
                    batchsize, 1, 'test')
