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
    LeNet-like convolutional neural network
    https://github.com/zalandoresearch/fashion-mnist/blob/master/benchmark/convnet.py
    """
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = th.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = th.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = th.nn.Linear(64*7*7, 1024)
        self.fc2 = th.nn.Linear(1024, 10)

    def forward(self, x):
        # -1, 1, 28, 28
        x = th.nn.functional.relu(self.conv1(x))
        # -1, 32, 28, 28
        x = th.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        # -1, 32, 14, 14
        x = th.nn.functional.relu(self.conv2(x))
        # -1, 64, 14, 14
        x = th.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        # -1, 64, 7, 7
        x = x.view(-1, 64*7*7)
        # -1, 64*7*7
        x = th.nn.functional.relu(self.fc1(x))
        x = th.nn.functional.dropout(x, p=0.4, training=self.training)
        # -1, 1024
        x = self.fc2(x)
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

    def loss_adversary(self, x, y, *, eps=0.0, maxiter=10):
        '''
        Train the network with PGD adversary
        '''
        images = x.to(self.fc1.weight.device)
        labels = y.to(self.fc1.weight.device).view(-1)
        images_orig = images.clone().detach()
        images.requires_grad = True

        # baseline: forward original samples
        with th.no_grad():
            self.eval()
            output, loss = self.loss(images, labels)
            acc_orig = output.max(1)[1].eq(labels).sum().item() / len(y)
            output_orig = output.clone().detach()
            loss_orig = loss.clone().detach()

        # start PGD attack
        alpha = eps / max(1, maxiter//2)
        for iteration in range(maxiter):
            self.train()
            optim = th.optim.SGD(self.parameters(), lr=1.)
            optimx = th.optim.SGD([images], lr=1.)
            optim.zero_grad(); optimx.zero_grad()
            output, loss = self.loss(images, labels)
            loss = -loss # gradient ascent
            loss.backward()

            images.grad.data.copy_(alpha * th.sign(images.grad))
            optimx.step()
            images = th.min(images, images_orig + eps)
            images = th.max(images, images_orig - eps)
            images = th.clamp(images, min=0., max=1.)
            images = images.clone().detach()
            images.requires_grad = True
            #print('> PGD internal loop', 'iteration=', iteration, 'loss=', loss.item())
        optim = th.optim.SGD(self.parameters(), lr=1.)
        optimx = th.optim.SGD([images], lr=1.)
        optim.zero_grad(); optimx.zero_grad()
        images.requires_grad = False

        # forward the PGD adversary
        output_adv, loss = common.cls_loss(self, images, labels, self.fc1.weight.device)
        acc_adv = output_adv.max(1)[1].eq(labels).sum().item() / len(y)
        print('* Orig loss', '%.5f'%loss_orig, 'Acc', '%.3f'%acc_orig,
                '\t|\t', '[Adv loss]', '%.5f'%loss.item(), '%.3f'%acc_adv)
        return (output_adv, loss)

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
