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
from .dbg import _fgcY


class Model(th.nn.Module):
    """
    Res18 + SOP dataset
    http://cvgl.stanford.edu/projects/lifted_struct/
    Although the original work uses Inception v? network
    """
    def setup_distance_metric(self, metric = 'C'):
        self.metric = metric

    def to(self, device):
        self.device = device
        return self._apply(lambda x: x.to(device))

    def __init__(self, finetune=True):
        super(Model, self).__init__()
        finetune = bool(int(os.getenv('FINETUNE', finetune)))
        if not finetune:
            print(_fgcY('Note, setting pretrain=False for ResNet18 as requested'))
        self.setup_distance_metric()
        self.res18 = vision.models.resnet18(pretrained=True if finetune else False)
        self.res18.fc = th.nn.Flatten() #th.nn.Linear(512, 512)
        self.finetune = finetune

    def train(self, mode=True):
        '''
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.train
        '''
        self.training = mode
        for module in self.children():
            module.train(mode)
        if not self.finetune:
            return self
        for part in (self.res18.conv1,
                self.res18.bn1,
                self.res18.layer1,
                self.res18.layer2,
                self.res18.layer3):
            for param in part.parameters():
                param.requires_grad = False
        return self

    def forward(self, x, *, l2norm=False):
        '''
        Input[x]: batch of images [N, 3, 224, 224]
        Output: representations [N, d]
        '''
        # -1, 3, 224, 224
        x = self.res18(x)
        # -1, 512
        if l2norm:
            x = x / x.norm(2, dim=1, keepdim=True).expand(*x.shape)
        return x

    def loss(self, x, y, *, marginC=0.5, marginE=1.0):
        '''
        Input[x]: batch of image pairs [2N, 3, 224, 224]
        Input[y]: "isSame" boolean labels [N]
        '''
        x = x.to(self.device).view(-1, 3, 224, 224)
        y = y.to(self.device).view(-1)
        output = self.forward(x)
        # compute loss function: pairwise hinge embedding loss
        if self.metric == "C": # Cosine
            loss = th.nn.functional.cosine_embedding_loss(
                    output[0::2,:], output[1::2,:], y.float(),
                    margin=marginC, reduction='sum')
        else: # Euclidean
            loss = th.nn.functional.hinge_embedding_loss(
                    th.nn.functional.pairwise_distance(
                        output[0::2], output[1::2]), y,
                    margin=marginE, reduction='sum')
        return output, loss

    def loss_adversary(self, x, y, *, eps=0.0, maxiter=10, hard=False):
        '''
        Train the network with a PGD adversary
        '''
        images = x.clone().detach().to(self.device).view(-1, 3, 224, 224)
        labels = y.to(self.device).view(-1)
        images_orig = images.clone().detach()
        images.requires_grad = True

        # preparation
        IMmean = th.tensor([0.485, 0.456, 0.406], device=self.device)
        IMstd = th.tensor([0.229, 0.224, 0.225], device=self.device)
        renorm = lambda im: im.sub(IMmean[:,None,None]).div(IMstd[:,None,None])
        denorm = lambda im: im.mul(IMstd[:,None,None]).add(IMmean[:,None,None])

        # first evaluation
        with th.no_grad():
            output, loss_orig = self.loss(images, labels)
            if self.metric == 'C':
                output /= output.norm(2, dim=1, keepdim=True).expand(*output.shape)
            output_orig = output.clone().detach()
            output_orig_nodetach = output

        # start PGD attack, we only move the anchor point
        alpha = float(min(max(eps/10., 1./255.), 0.01))  # PGD hyper-parameter
        maxiter = int(min(max(10, 2*eps/alpha), 30))  # PGD hyper-parameter

        # PGD with/without random init?
        if int(os.getenv('RINIT', 0))>0:
            images = images + (eps/IMstd[:,None,None])*2*(0.5-th.rand(images.shape)).to(images.device)
            images = th.max(images, renorm(th.zeros(images.shape).to(device)))
            images = th.min(images, renorm(th.ones(images.shape).to(device)))
            images = images.detach()
            images.requires_grad = True

        for iteration in range(maxiter):
            self.train()
            optim = th.optim.SGD(self.parameters(), lr=1.)
            optimx = th.optim.SGD([images], lr=1.)
            optim.zero_grad(); optimx.zero_grad()

            # do we only attack the anchor?
            USE_STRIPE=False # STRIPE: a~, p, n; NO-STRIPE: a~,p~,n~
            stripe = 3 if images.shape[0]%3==0 else 2
            if USE_STRIPE and self.metric == 'C':
                output = self.forward(images[::stripe], l2norm=True)
                distance = 1 - th.mm(output, output_orig[::stripe].t())
                loss = -distance.trace() # gradient ascent on trace, i.e. diag.sum
            elif (not USE_STRIPE) and self.metric == 'C':
                output = self.forward(images, l2norm=True)
                distance = 1 - th.mm(output, output_orig.t())
                loss = -distance.trace()
            elif USE_STRIPE and self.metric == 'E':
                output = self.forward(images[::stripe], l2norm=False)
                distance = th.nn.functional.pairwise_distance(
                        output, output_orig[::stripe], p=2)
                loss = -distance.sum()
            elif (not USE_STRIPE) and self.metric == 'E':
                output = self.forward(images, l2norm=False)
                distance = th.nn.functional.pairwise_distance(
                        output, output_orig, p=2)
                loss = -distance.sum()
            loss.backward()

            # PGD for normalized images is a little bit special
            images.grad.data.copy_((alpha/IMstd[:,None,None])*th.sign(images.grad))
            optimx.step()
            images = th.min(images, images_orig + (eps/IMstd[:,None,None]))
            images = th.max(images, images_orig - (eps/IMstd[:,None,None]))
            #images = th.clamp(images, min=0., max=1.)
            images = th.max(images, renorm(th.zeros(images.shape).to(self.device)))
            images = th.min(images, renorm(th.ones(images.shape).to(self.device)))
            images = images.clone().detach()
            images.requires_grad = True
            #print('> Internal PGD loop [', iteration, ']', 'loss=', loss.item())
        optim = th.optim.SGD(self.parameters(), lr=1.)
        optimx = th.optim.SGD([images], lr=1.)
        optim.zero_grad(); optimx.zero_grad()
        images.requires_grad = False

        # forward the adversarial example
        if False:
            #== trip-es defense
            _, loss_adv = self.loss(images_orig, labels)
            if self.metric == 'C':
                output = self.forward(images, l2norm=True)
                raise NotImplementedError
            elif self.metric == 'E':
                output = self.forward(images, l2norm=False)
                loss_es = th.nn.functional.pairwise_distance(output, output_orig_nodetach, p=2).mean()
                loss_adv = loss_adv + 1.0 * loss_es
            print('* Orig loss', '%.5f'%loss_orig.item(), '  |  ',
                    '[Adv loss]', '%.5f'%loss_adv.item(), '\twhere loss_ES=', loss_es.item())
            return output, loss_adv
        else:
            #== min(Trip(max(ES))) defense
            output, loss_adv = self.loss(images, labels)
            print('* Orig loss', '%.5f'%loss_orig.item(), '  |  ',
                    '[Adv loss]', '%.5f'%loss_adv.item())
            return output, loss_adv

    def report(self, epoch, iteration, total, output, labels, loss):
        print(f'Eph[{epoch}][{iteration}/{total}]',
                collections.namedtuple('Res', ['loss'])('%.4f'%loss.item()))

    def validate(self, dataloader):
        self.eval()

        # gather the embedding vectors
        allRepr, allLabel = common.compute_embedding(self, dataloader,
                l2norm=(True if self.metric == 'C' else False),
                device=self.device)
        allLabel = allLabel.cpu().squeeze().numpy()

        # calculate the per-example recall
        r_mean = []
        r_1, r_10, r_100, r_1000 = [], [], [], []
        with th.no_grad():
            for i in tqdm(range(allRepr.shape[0])):
                xq = allRepr[i].view(1, -1)  # [1,512]
                yq = allLabel[i]
                if self.metric == 'C': # Cosine
                    mpdist = (1 - th.mm(allRepr, xq.t())).cpu().numpy().squeeze()
                else: # Euclidean
                    mpdist = (allRepr - xq).norm(dim=1).cpu().numpy().squeeze()
                agsort = mpdist.argsort()[1:]
                rank = np.where(allLabel[agsort] == yq)[0].min()
                r_mean.append(rank)
                r_1.append(rank == 0)
                r_10.append(rank < 10)
                r_100.append(rank < 100)
                r_1000.append(rank < 1000)
        r_mean = np.mean(r_mean)
        r_1 = np.mean(r_1)
        r_10 = np.mean(r_10)
        r_100 = np.mean(r_100)
        r_1000 = np.mean(r_1000)

        return f'r@mean {r_mean:.1f} (/{allRepr.size(0)}) r@1 {r_1:.3f} (/1)' + \
                f'r@10 {r_10:.3f} (/1) r@100 {r_100:.3f} (/1) r@1000 {r_1000:.3f} (/1)'

    def getloader(self, kind:str='train', batchsize:int=1):
        '''
        get corresponding dataloaders
        '''
        config = yaml.load(open('config.yml', 'r').read(),
                Loader=yaml.SafeLoader)
        if kind == 'train':
            return datasets.sop.get_loader(
                    os.path.expanduser(config['sop']['path']),
                    batchsize, 2, 'train')
        else:
            return datasets.sop.get_loader(
                    os.path.expanduser(config['sop']['path']),
                    batchsize, 1, 'test')

    def attack(self, att, loader, *, dconf, verbose=False):
        device = self.device
        dconf['metric'] = self.metric
        dconf['normimg'] = True
        dconf['device'] = self.device
        #if 'PGD-UT' in att:
        #    return common.QA_PGD_UT(self, loader, dconf=dconf, verbose=verbose)
        return common.rank_attack(self, att, loader,
                dconf=dconf, device=device, verbose=verbose)
