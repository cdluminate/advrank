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
import numpy as np
from . import datasets, common

class Model(th.nn.Module):
    """
    LeNet-like convolutional neural network for embedding
    https://github.com/zalandoresearch/fashion-mnist/blob/master/benchmark/convnet.py
    """
    def setup_distance_metric(self, metric='C'):
        self.metric = 'C'

    def to(self, device):
        self.device = device
        return self._apply(lambda x: x.to(device))

    def __init__(self):
        super(Model, self).__init__()
        self.setup_distance_metric()
        self.conv1 = th.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = th.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = th.nn.Linear(64*7*7, 1024)

    def forward(self, x, *, l2norm=False):
        '''
        Input[x]: tensor in shape [N, 1, 28, 28]
        Output: tensor in shape [N, d]
        '''
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
        x = self.fc1(x)
        # -1, 1024
        if l2norm:
            x = x / x.norm(2, dim=1, keepdim=True).expand(*x.shape)
        return x

    def loss(self, x, y, *, marginC=0.2, marginE=1.0, hard=False):
        '''
        Input[x]: tensor in shape [N, 1, 28, 28]
        Input[y]: tensor in shape [N], class labels ranging from 1-10
        '''
        x = x.to(self.fc1.weight.device)
        y = y.to(self.fc1.weight.device).view(-1)
        output = self.forward(x)
        if self.metric == 'C': # Cosine
            # normalize to unit L-2 norm and prepare label matrix
            X = output / output.norm(2, dim=1, keepdim=True).expand(
                    *(output.shape))
            Y = y.unsqueeze(0).expand(len(y), len(y))
            Y = Y.eq(Y.t())
            # compute loss function: pairwise hinge embedding loss
            dist = 1 - X @ X.t()
            loss = Y.eq(1).float() * dist + \
                Y.ne(1).float() * (marginC - dist).clamp(min=0.)
            if not hard:
                loss = loss.mean()
            else:
                loss = loss.max(1)[0].mean()
        elif self.metric == 'E': # Euclidean
            # -- exhaustive matrix version
            # we can also write a random pair verison: idx=th.randint(0,...)
            Y = y.unsqueeze(0).expand(len(y), len(y))
            Y = Y.eq(Y.t())
            M, K = output.shape[0], output.shape[1]
            pdist = (output.view(M,1,K).expand(M,M,K) - \
                    output.view(1,M,K).expand(M,M,K)).norm(dim=2)
            loss = Y.eq(1).float() * pdist + \
                    Y.ne(1).float() * (marginE - pdist).clamp(min=0.)
            if not hard:
                loss = loss.mean()
            else:
                loss = loss.max(1)[0].mean()
        else:
            raise ValueError(self.metric)
        return output, loss

    def loss_adversary(self, x, y, *, eps=0.0, maxiter=10, hard=False, marginC=0.2, marginE=1.0):
        '''
        Train the network with a PGD adversary
        '''
        images = x.to(self.fc1.weight.device)
        labels = y.to(self.fc1.weight.device).view(-1)
        images_orig = images.clone().detach()
        images.requires_grad = True

        # first forwarding
        with th.no_grad():
            output, loss_orig = self.loss(images, labels)
            if self.metric == 'C':
                output /= output.norm(2, dim=1, keepdim=True).expand(*output.shape)
            elif self.metric == 'E':
                pass
            output_orig = output.clone().detach()
            output_orig_nodetach = output

        # start PGD attack
        alpha = float(min(max(eps/10., 1./255.), 0.01))  # PGD hyper-parameter
        maxiter = int(min(max(10, 2*eps/alpha), 30))  # PGD hyper-parameter

        # PGD with/without random init?
        if int(os.getenv('RINIT', 0))>0:
            images = images + eps*2*(0.5-th.rand(images.shape)).to(images.device)
            images = th.clamp(images, min=0., max=1.)
            images = images.detach()
            images.requires_grad = True

        for iteration in range(maxiter):
            self.train()
            optim = th.optim.SGD(self.parameters(), lr=1.)
            optimx = th.optim.SGD([images], lr=1.)
            optim.zero_grad(); optimx.zero_grad()

            if self.metric == 'C':
                output = self.forward(images, l2norm=True)
                distance = 1 - th.mm(output, output_orig.t())
                loss = -distance.trace() # gradient ascent on trace, i.e. diag.sum
            elif self.metric == 'E':
                output = self.forward(images, l2norm=False)
                distance = th.nn.functional.pairwise_distance(
                        output, output_orig, p=2)
                loss = -distance.sum()
            else:
                raise ValueError(self.metric)
            loss.backward()

            images.grad.data.copy_(alpha * th.sign(images.grad))
            optimx.step()
            images = th.min(images, images_orig + eps)
            images = th.max(images, images_orig - eps)
            images = th.clamp(images, min=0., max=1.)
            images = images.clone().detach()
            images.requires_grad = True
            #print('> Internal PGD loop [', iteration, ']', 'loss=', loss.item())

        # XXX: It's very critical to clear the junk gradients
        optim = th.optim.SGD(self.parameters(), lr=1.)
        optimx = th.optim.SGD([images], lr=1.)
        optim.zero_grad(); optimx.zero_grad()
        images.requires_grad = False

        # forward the adversarial example
        if False:
            #== trip-es loss
            _, loss_adv = self.loss(images_orig, labels)
            if self.metric == 'C':
                output = self.forward(images, l2norm=True)
                loss_es = (1 - th.mm(output, output_orig_nodetach.t())).trace() / output.size(0)
                loss_adv = loss_adv + 1.0 * loss_es
            elif self.metric == 'E':
                output = self.forward(images, l2norm=False)
                loss_es = th.nn.functional.pairwise_distance(output, output_orig_nodetach, p=2).mean()
                loss_adv = loss_adv + 1.0 * loss_es  # very unstable
            print('* Orig loss', '%.5f'%loss_orig.item(), '\t|\t',
                    '[Adv loss]', '%.5f'%loss_adv.item(), '\twhere loss_ES=', loss_es.item())
            return output, loss_adv
        else:
            #== min(Trip(max(ES(\tilde(a))))) loss
            output, loss_adv = self.loss(images, labels)
            print('* Orig loss', '%.5f'%loss_orig.item(), '\t|\t',
                    '[Adv loss]', '%.5f'%loss_adv.item())
            return output, loss_adv

    def report(self, epoch, iteration, total, output, labels, loss):
        if self.metric == 'C': # Cosine
            X = output / output.norm(2, dim=1, keepdim=True).expand(
                    *(output.shape))
            dist = 1 - X @ X.t()
            offdiagdist = dist + math.e*th.eye(X.shape[0]).to(X.device)
            knnsearch = labels[th.min(offdiagdist, dim=1)[1]].cpu()
            acc = 100.*knnsearch.eq(labels.cpu()).sum().item() / len(labels)
        elif self.metric == 'E': # Euclidean
            M, K = output.shape[0], output.shape[1]
            pdist = (output.view(M,1,K).expand(M,M,K) - \
                    output.view(1,M,K).expand(M,M,K)).norm(dim=2)
            offdiagdist = pdist + 1e9*th.eye(pdist.shape[0]).to(output.device)
            knnsearch = labels[th.min(offdiagdist, dim=1)[1]].cpu()
            acc = 100.*knnsearch.eq(labels.cpu()).sum().item() / len(labels)
        else:
            raise ValueError(self.metric)
        print(f'Eph[{epoch}][{iteration}/{total}]',
                collections.namedtuple('Res', ['loss', 'r_1'])(
                    '%.4f'%loss.item(), '%.3f'%acc))

    def validate(self, dataloader):
        self.eval()
        if self.metric == 'C': # Cosine
            allFeat, allLab = common.compute_embedding(self, dataloader,
                    l2norm=True, device=self.device)
            dist = 1 - th.mm(allFeat, allFeat.t())
            offdiagdist = dist + \
                    math.e * th.eye(allFeat.shape[0]).to(allFeat.device)
            knnsearch = allLab[th.min(offdiagdist, dim=1)[1]].cpu()
            correct = knnsearch.eq(allLab.cpu()).sum().item()
            total = len(allLab)
            result = f'r@1 = {100.*correct/total} (/100)'
        else: # Euclidean
            allEmb, allLab = common.compute_embedding(self, dataloader,
                    l2norm=False, device=self.device)
            correct = []
            for i in range(allEmb.shape[0]):
                pdist = (allEmb - allEmb[i].view(1, allEmb.shape[1])).norm(dim=1)
                knnsrch = allLab[pdist.topk(k=2,largest=False)[1][1]]
                correct.append((knnsrch == allLab[i]).item())
            correct = np.sum(correct)
            total = len(allLab)
            result = f'r@1= {100.*correct/total} (/100)'
        return result

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

    def attack(self, att, loader, *, dconf, verbose=False):
        device = self.fc1.weight.device
        dconf['metric'] = self.metric
        return common.rank_attack(self, att, loader,
                dconf=dconf, device=device, verbose=verbose)
