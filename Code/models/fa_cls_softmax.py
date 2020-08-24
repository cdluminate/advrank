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
from . import datasets, common
import yaml
from tqdm import tqdm


class Model(common.Model):
    """
    Softmax + cross entropy
    """
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = th.nn.Linear(28*28, 10)

    def forward(self, x):
        # -1, 1, 28, 28
        x = x.view(-1, 28*28)
        # -1, 28*28
        x = self.fc1(x)
        return x

    def getloader(self, kind:str='train', batchsize:int=1):
        '''
        get corresponding dataloaders
        '''
        config = yaml.load(open('config.yml', 'r').read(),
                Loader=yaml.SafeLoader)
        if kind == 'train':
            return datasets.fashion.get_loader(
                    os.path.expanduser(config['fashion-mnist']['path']),
                    batchsize, 'train')
        else:
            return datasets.fashion.get_loader(
                    os.path.expanduser(config['fashion-mnist']['path']),
                    batchsize, 't10k')

    def loss(self, x, y):
        device = self.fc1.weight.device
        return common.cls_loss(self, x, y, device)

    def report(self, epoch, iteration, total, output, labels, loss):
        result = common.cls_report(output, labels, loss)
        print(f'Eph[{epoch}][{iteration}/{total}]', result)

    def validate(self, dataloader):
        device = self.fc1.weight.device
        return common.cls_validate(self, dataloader)

    def attack(self, att, loader, *, dconf, verbose=False):
        device = self.fc1.weight.device
        return common.cls_attack(self, att, loader,
                dconf=dconf, device=device, verbose=verbose)
