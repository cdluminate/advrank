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
import os, sys, re
import functools
import torch as th
import collections
from tqdm import tqdm
import pylab as lab
import traceback
import math
import statistics
from scipy import stats
import numpy as np
from .dbg import printStack, _fgcG, _bgcV, _fgcGrey, _fgcCyan
import random

#######################################################################

IMmean = th.tensor([0.485, 0.456, 0.406])
IMstd = th.tensor([0.229, 0.224, 0.225])
renorm = lambda im: im.sub(IMmean[:,None,None]).div(IMstd[:,None,None])
denorm = lambda im: im.mul(IMstd[:,None,None]).add(IMmean[:,None,None])
xdnorm = lambda im: im.div(IMstd[:,None,None]).add(IMmean[:,None,None])

#######################################################################

class Model(th.nn.Module):
    '''
    Base model. May be a classification model, or a embedding/ranking model.
    '''
    def setup_distance_metric(self, metric):
        '''
        Do we use Cosine distance or Euclidean distance on
        the embedding space?
        '''
        raise NotImplementedError

    def forward(self, x):
        '''
        Purely input -> output, no loss function
        '''
        raise NotImplementedError

    def loss(self, x, y, device='cpu'):
        '''
        Combination: input -> output -> loss
        '''
        raise NotImplementedError

    def loss_adversary(self, x, y, device='cpu', *, eps=0.0):
        '''
        Adversarial training: replace normal example with adv example
        https://github.com/MadryLab/mnist_challenge/blob/master/train.py
        '''
        raise NotImplementedError

    def report(self, epoch, iteration, total, output, labels, loss):
        '''
        Given the (output, loss) combination, report current stat
        '''
        raise NotImplementedError

    def validate(self, dataloader, device='cpu'):
        '''
        Run validation on the given dataset
        '''
        raise NotImplementedError

    def getloader(self, kind:str='train', batchsize:int=1):
        '''
        Load the specific dataset for the model
        '''
        raise NotImplementedError

    def attack(self, attack, loader, *, dconf, device, verbose=False):
        '''
        Apply XXX kind of attack
        '''
        raise NotImplementedError

#######################################################################

def cls_loss(model, x, y, device):
    x = x.to(device)
    y = y.to(device).view(-1)
    output = model.forward(x)
    loss = th.nn.functional.cross_entropy(output, y)
    return output, loss

def cls_report(output, labels, loss):
    pred = output.max(1)[1].cpu()
    acc = 100.*pred.eq(labels.cpu().flatten()).sum().item() / len(labels)
    return collections.namedtuple(
            'Res', ['loss', 'accu'])('%.2f'%loss.item(), '%.2f'%acc)

def cls_validate(model, dataloader):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with th.no_grad():
        for iteration, (images, labels) in enumerate(dataloader):
            labels = labels.view(-1)
            output, loss = model.loss(images, labels)
            test_loss += loss.item()
            pred = output.max(1)[1].cpu().view(-1)
            correct += pred.eq(labels.cpu().flatten()).sum().item()
            total += len(labels)
    return collections.namedtuple('Result', ['loss', 'accuracy'])(
            test_loss, 100 * correct / total)


def cls_attack(model, attack, loader, *, dconf, device, verbose=False):
    '''
    generic attack method for classification models
    '''
    orig_correct, adv_correct, total = 0, 0, 0
    for N, (images, labels) in tqdm(enumerate(loader)):
        if attack == 'ES:FGSM-UT':
            xr, r, out, loss, count = ClassPGD(model, images, labels,
                eps=dconf['epsilon'], verbose=verbose, device=device, maxiter=1)
        elif attack == 'ES:PGD-UT':
            xr, r, out, loss, count = ClassPGD(model, images, labels,
                eps=dconf['epsilon'], verbose=verbose, device=device, maxiter=10)
        elif attack.startswith('C:'):
            raise ValueError(f'No candidate attack for classifier.')
        else:
            raise ValueError(f"Attack {attack} unsupported.")
        orig_correct += count[0]
        adv_correct += count[1]
        total += len(labels)
    print('baseline=', '%.2f'%(100.*(orig_correct/total)),
            'adv=', '%.2f'%(100.*(adv_correct/total)),
            'advReduce=', '%.2f'%(100.*(orig_correct - adv_correct) / total))


def ClassPGD(model, images, labels, *, eps, alpha=None, maxiter=1, verbose=False, device='cpu'):
    '''
    Perform FGSM/PGD attack on the given batch of images L_infty constraint
    https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks/fast_gradient_method.py
    https://github.com/bethgelab/foolbox/blob/master/foolbox/attacks/gradient.py
    https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks/projected_gradient_descent.py
    '''
    assert(type(images) == th.Tensor)

    images = images.to(device)
    images_orig = images.clone().detach()
    images.requires_grad = True
    labels = labels.to(device).view(-1)

    # baseline: forward original samples
    model.eval()
    with th.no_grad():
        output, loss = model.loss(images, labels)
        orig_correct = output.max(1)[1].eq(labels).sum().item()
        output_orig = output.clone().detach()
        loss_orig = loss.clone().detach()
    if verbose:
        print()
        print('* Original Sample', 'loss=', loss.item(), 'acc=', orig_correct/len(labels))
        #print('> labels=', [x.item() for x in labels])
        #print('> predic=', [x.item() for x in output.max(1)[1]])

    # start attack
    alpha = eps / max(1, maxiter//2)
    for iteration in range(maxiter):
        model.train()
        optim = th.optim.SGD(model.parameters(), lr=1.)
        optimx = th.optim.SGD([images], lr=1.)
        optim.zero_grad(); optimx.zero_grad()
        output, loss = model.loss(images, labels)
        loss = -loss # gradient ascent
        loss.backward()

        if maxiter>1:
            images.grad.data.copy_(alpha * th.sign(images.grad))
        else:
            images.grad.data.copy_(eps * th.sign(images.grad))
        optimx.step()
        images = th.min(images, images_orig + eps)
        images = th.max(images, images_orig - eps)
        images = th.clamp(images, min=0., max=1.)
        images = images.clone().detach()
        images.requires_grad = True
        if maxiter>1:
            print('iteration', iteration, 'loss', loss.item())
    xr = images
    r = images - images_orig

    # adversarial: forward perturbed images
    model.eval()
    with th.no_grad():
        output, loss = model.loss(images, labels)
        adv_correct = output.max(1)[1].eq(labels).sum().item()
    if verbose:
        print('* Adversarial example', 'loss=', loss.item(), 'acc=', adv_correct/len(labels))
        #print('> labels=', [x.item() for x in labels])
        #print('> predic=', [x.item() for x in output.max(1)[1]])
        print('r>', 'Min', '%.3f'%r.min().item(),
                'Max', '%.3f'%r.max().item(),
                'Mean', '%.3f'%r.mean().item(),
                'L0', '%.3f'%r.norm(0).item(),
                'L1', '%.3f'%r.norm(1).item(),
                'L2', '%.3f'%r.norm(2).item())
        # print the image if batchsize is 1
        if xr.shape[0] == 1:
            im = images_orig.cpu().squeeze().view(28, 28).numpy()
            advim = xr.cpu().squeeze().view(28, 28).numpy()
            advpr = r.cpu().squeeze().view(28, 28).numpy()
            lab.subplot(131)
            lab.imshow(im, cmap='gray')
            lab.colorbar()
            lab.subplot(132)
            lab.imshow(advpr, cmap='gray')
            lab.colorbar()
            lab.subplot(133)
            lab.imshow(advim, cmap='gray')
            lab.colorbar()
            lab.show()

    return xr, r, (output_orig, output), \
            (loss_orig, loss), (orig_correct, adv_correct)


###########################################################################


def compute_embedding(model, dataloader, *, l2norm=False, device='cpu'):
    '''
    Compute the embedding vectors
    '''
    model.eval()
    allEmb, allLab = [], []
    if int(os.getenv('EMBCACHE', 0))>0 and os.path.exists('trained/embcache.th'):
        print('Loading embedding cache ...')
        allEmb, allLab = th.load('trained/embcache.th')
        print('allEmb.shape=', allEmb.shape, 'allLab.shape=', allLab.shape)
        return (allEmb, allLab)
    with th.no_grad():
        for iteration, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
            images, labels = images.to(device), labels.to(device)
            labels = labels.view(-1)
            output = model.forward(images)
            if l2norm: # do L-2 normalization
                output /= output.norm(
                        2, dim=1, keepdim=True).expand(*(output.shape))
            allEmb.append(output.detach())
            allLab.append(labels)
            if int(os.getenv('EMBBREAK', 0)) > 0:
                if int(os.getenv('EMBBREAK')) <= iteration:
                    break # debug
    allEmb, allLab = th.cat(allEmb), th.cat(allLab)
    if int(os.getenv('EMBCACHE', 0))>0:
        if not os.path.exists('trained/embcache.th'):
            print('Saving embedding cache ...')
            th.save([allEmb.detach().cpu(), allLab.detach().cpu()], 'trained/embcache.th')
    return allEmb, allLab


def rank_attack(model, attack, loader, *, dconf, device, verbose=False):
    '''
    generic attack method for embedding/ranking models
    '''
    # >> pre-process the options
    normimg = dconf.get('normimg', False)
    if dconf.get('metric', None) is None:
        raise ValueError('dconf parameter misses the "metric" key')
    candidates = compute_embedding(model, loader, device=device,
            l2norm=(True if dconf['metric']=='C' else False))
    if dconf.get('TRANSFER', None) is None:
        dconf['TRANSFER'] = None
    else:
        candidates_trans = compute_embedding(dconf['TRANSFER']['model'],
                loader, device=dconf['TRANSFER']['device'],
                l2norm=(True if 'C' in dconf['TRANSFER']['transfer'] else False))
        dconf['TRANSFER']['candidates'] = candidates_trans
    ruthless = int(os.getenv('RUTHLESS', -1))  # maxiter for attack

    # >> dispatch: attacking
    print('>>> Candidate Set:', candidates[0].shape, candidates[1].shape)
    correct_orig, correct_adv, total = 0, 0, 0
    rankup, embshift, prankgt, prank_trans = [], [], [], []
    for N, (images, labels) in tqdm(enumerate(loader)):
        #if N < random.randint(0, 60502//2): continue # picking sample for vis
        #if N < 14676//2: continue
        if (ruthless > 0) and (N >= ruthless):
            break
        if verbose: print(_fgcCyan('\n'+'\u2500'*64))
        if attack == 'ES:PGD-UT':
            # if you set the PGD iteration to 1, the attack will degenerate
            # into ES:FGSM-UT
            xr, r, out, loss, count = RankPGD(model, images, labels,
                    candidates, eps=dconf['epsilon'], verbose=verbose,
                    device=device, loader=loader, metric=dconf['metric'],
                    normimg=normimg, atype='ES',
                    transfer=dconf['TRANSFER'])
        elif re.match('^C.?:PGD-W\d+$', attack) is not None:
            regroup = re.match('^C(.?):PGD-W(\d+)$', attack).groups()
            pm = str(regroup[0]) # + / -
            assert(pm in ['+', '-'])
            W = int(regroup[1]) # w, num of queries
            assert(W > 0)
            xr, r, out, loss, count = RankPGD(model, images, labels,
                    candidates, eps=dconf['epsilon'], verbose=verbose,
                    device=device, loader=loader, metric=dconf['metric'],
                    normimg=normimg, atype='CA', W=W, pm=pm,
                    transfer=dconf['TRANSFER'])
        elif re.match('^Q.?:PGD-M\d+$', attack) is not None:
            regroup = re.match('^Q(.?):PGD-M(\d+)$', attack).groups() 
            pm = str(regroup[0]) # + / -
            assert(pm in ['+', '-'])
            M = int(regroup[1]) # m, num of candidates
            assert(M > 0)
            xr, r, out, loss, count = RankPGD(model, images, labels,
                    candidates, eps=dconf['epsilon'], verbose=verbose,
                    device=device, loader=loader, metric=dconf['metric'],
                    normimg=normimg, atype='QA', M=M, pm=pm,
                    transfer=dconf['TRANSFER'])
        elif re.match('^SPQ.?:PGD-M\d+$', attack) is not None:
            regroup = re.match('^SPQ(.?):PGD-M(\d+)$', attack).groups()
            pm = str(regroup[0]) # + / -
            assert(pm in ['+', '-'])
            M = int(regroup[1]) # m, num of candidates
            assert(M > 0)
            xr, r, out, loss, count = RankPGD(model, images, labels,
                    candidates, eps=dconf['epsilon'], verbose=verbose,
                    device=device, loader=loader, metric=dconf['metric'],
                    normimg=normimg, atype='SPQA', M=M, pm=pm,
                    transfer=dconf['TRANSFER'])
        else:
            raise ValueError(f"Attack {attack} unsupported.")
        correct_orig += count[0][0]
        correct_adv += count[1][0]
        total += len(labels)
        rankup.append(count[1][1])
        embshift.append(count[1][2])
        prankgt.append(count[1][3])
        prank_trans.append(count[1][4])
    # >> report overall attacking result on the test dataset
    print(_fgcCyan('\u2500'*64))
    if int(os.getenv('IAP', 0)) > 0:
        print(_fgcCyan(f'Summary[{attack} \u03B5={dconf["epsilon"]}]:',
                'white-box=', '%.3f'%statistics.mean(rankup), # abuse var
                'black-box=', '%.3f'%statistics.mean(embshift), # abuse var
                'white-box-orig=', '%.3f'%statistics.mean(prankgt), # abuse var
                'black-box-orig=', '%.3f'%statistics.mean(prank_trans), # abuse var
                ))
    else:
        print(_fgcCyan(f'Summary[{attack} \u03B5={dconf["epsilon"]}]:',
                'baseline=', '%.3f'%(100.*(correct_orig/total)),
                'adv=', '%.3f'%(100.*(correct_adv/total)),
                'advReduce=', '%.3f'%(100.*(correct_orig - correct_adv) / total),
                'rankUp=', '%.3f'%statistics.mean(rankup),
                'embShift=', '%.3f'%statistics.mean(embshift),
                'prankgt=', '%.3f'%statistics.mean(prankgt),
                'prank_trans=', '%.3f'%statistics.mean(prank_trans),
                ))
    print(_fgcCyan('\u2500'*64))


class LossFactory(object):
    '''
    Factory of loss functions used in all ranking attacks
    '''
    @staticmethod
    def RankLossEmbShift(repv: th.tensor, repv_orig: th.tensor, *, metric: str):
        '''
        Computes the embedding shift, we want to maximize it by gradient descent
        '''
        if metric == 'C':
            distance = 1 - th.mm(repv, repv_orig.t())
            loss = -distance.trace() # gradient ascent on trace, i.e. diag.sum
        elif metric == 'E':
            distance = th.nn.functional.pairwise_distance(repv, repv_orig, p=2)
            loss = -distance.sum()
        return loss
    @staticmethod
    def RankLossQueryAttack(qs: th.tensor, Cs: th.tensor, Xs: th.tensor, *, metric: str, pm: str,
            dist: th.tensor = None, cidx: th.tensor = None):
        '''
        Computes the loss function for pure query attack
        '''
        assert(qs.shape[1] == Cs.shape[2] == Xs.shape[1])
        NIter, M, D, NX = qs.shape[0], Cs.shape[1], Cs.shape[2], Xs.shape[0]
        DO_RANK = (dist is not None) and (cidx is not None)
        losses, ranks = [], []
        #refrank = []
        for i in range(NIter):
            #== compute the pairwise loss
            q = qs[i].view(1, D)  # [1, output_1]
            C = Cs[i, :, :].view(M, D)  # [1, output_1]
            if metric == 'C':
                A = (1 - th.mm(q, C.t())).expand(NX, M)
                B = (1 - th.mm(Xs, q.t())).expand(NX, M)
            elif metric == 'E':
                A = (C - q).norm(2, dim=1).expand(NX, M)
                B = (Xs - q).norm(2, dim=1).view(NX, 1).expand(NX, M)
            #== loss function
            if '+' == pm:
                loss = (A-B).clamp(min=0.).mean()
            elif '-' == pm:
                loss = (-A+B).clamp(min=0.).mean()
            losses.append(loss)
            #== compute the rank
            if DO_RANK:
                ranks.append(th.mean(dist[i].flatten().argsort().argsort()
                    [cidx[i,:].flatten()].float()).item())
            #refrank.append( ((A>B).float().mean()).item() )
        #print('(debug)', 'rank=', statistics.mean(refrank))
        loss = th.stack(losses).mean()
        rank = statistics.mean(ranks) if DO_RANK else None
        return loss, rank
    @staticmethod
    def RankLossImAgQA(qs: th.tensor, C: th.tensor, Xs: th.tensor, *, metric, pm):
        assert(qs.shape[1] == C.shape[1] == Xs.shape[1])
        N, D, NX = qs.size(0), C.shape[1], Xs.shape[0]
        # qs [N, D] C [1, D] Xs [NX, D]
        if metric == 'C':
            A = (1 - th.mm(C, qs.t())).view(1, N).expand(NX, N)
            B = (1 - th.mm(Xs, qs.t()))  # NX, N
        else:
            A = (qs - C).norm(2, dim=1).view(1, N).expand(NX, N)
            B = (Xs.view(NX,1,D).expand(NX,N,D) - \
                    qs.view(1,N,D).expand(NX,N,D)).norm(2,dim=2) # [NX,N] Fast but OOM
            #B = []
            #for _q in qs:
            #    _B = (Xs - _q).norm(2,dim=1).view(-1)
            #    B.append(_B)
            #B = th.stack(B).t() # [NX,N] memory efficient but slow
        if '+' == pm:
            loss = (A-B).clamp(min=0.).mean()
        else:
            loss = (-A+B).clamp(min=0.).mean()
        rank = ((A > B).float().mean()).item()  # normalized result
        return loss, rank
    @staticmethod
    def RankLossQueryAttackDistance(qs: th.tensor, Cs: th.tensor, Xs: th.tensor, *,
            metric: str, pm: str, dist: th.tensor = None, cidx: th.tensor = None):
        '''
        Actually the distance based objective is inferior
        '''
        N, M, D, NX = qs.shape[0], Cs.shape[1], Cs.shape[2], Xs.shape[0]
        assert(qs.shape[1] == Cs.shape[2] == Xs.shape[1]) # D
        losses, ranks = [], []
        for i in range(N):
            q = qs[i].view(1, D)
            C = Cs[i,:,:].view(M, D)
            if (metric, pm) == ('C', '+'):
                loss = (1 - th.mm(q, C.t())).mean()
            elif (metric, pm) == ('C', '-'):
                loss = -(1 - th.mm(q, C.t())).mean()
            elif (metric, pm) == ('E', '+'):
                loss = (C - q).norm(2, dim=1).mean()
            elif (metric, pm) == ('E', '-'):
                loss = -(C - q).norm(2, dim=1).mean()
            losses.append(loss)
            if metric == 'C':
                A = (1 - th.mm(q, C.t())).expand(NX, M)
                B = (1 - th.mm(Xs, q.t())).expand(NX, M)
            elif metric == 'E':
                A = (C - q).norm(2, dim=1).expand(NX, M)
                B = (Xs - q).norm(2, dim=1).view(NX, 1).expand(NX, M)
            rank = ((A > B).float().mean() * NX).item()  # non-normalized result
            ranks.append(rank)
        loss = th.stack(losses).mean()
        rank = statistics.mean(ranks)
        return (loss, rank)
    @staticmethod
    def RankLossCandidateAttack(cs: th.tensor, Qs: th.tensor, Xs: th.tensor, *, metric: str, pm: str):
        '''
        Computes the loss function for pure candidate attack
        '''
        assert(cs.shape[1] == Qs.shape[2] == Xs.shape[1])
        NIter, W, D, NX = cs.shape[0], Qs.shape[1], Qs.shape[2], Xs.shape[0]
        losses, ranks = [], []
        for i in range(NIter):
            #== compute pairwise distance
            c = cs[i].view(1, D) # [1, output_1]
            Q = Qs[i, :, :].view(W, D) # [W, output_1]
            if metric == 'C':
                A = 1 - th.mm(c, Q.t()).expand(NX, W) # [candi_0, W]
                B = 1 - th.mm(Xs, Q.t()) # [candi_0, W]
            elif metric == 'E':
                A = (Q - c).norm(2,dim=1).expand(NX, W) # [candi_0, W]
                B = (Xs.view(NX,1,D).expand(NX,W,D) - \
                        Q.view(1,W,D).expand(NX,W,D)).norm(2,dim=2) # [candi_0, W]
            #== loss function
            if '+' == pm:
                loss = (A-B).clamp(min=0.).mean()
            elif  '-' == pm:
                loss = (-A+B).clamp(min=0.).mean()
            losses.append(loss)
            #== compute the rank
            rank = ((A > B).float().mean() * NX).item()  # the > sign is correct
            ranks.append(rank)
        loss = th.stack(losses).mean()
        rank = statistics.mean(ranks)
        return loss, rank
    @staticmethod
    def RankLossImAgCA(cs: th.tensor, Q: th.tensor, Xs: th.tensor, *, metric, pm):
        assert(cs.shape[1] == Q.shape[1] == Xs.shape[1])
        N, D, NX = cs.size(0), Q.shape[1], Xs.shape[0]
        # cs [N, D] Q [1, D] Xs [NX, D]
        if metric == 'C':
            A = (1 - th.mm(cs, Q.view(D, 1))).view(N, 1).expand(N, NX)
            B = (1 - th.mm(Xs, Q.view(D, 1))).view(1, NX).expand(N, NX)
        else:
            A = (cs - Q).norm(2,dim=1).view(N, 1).expand(N, NX)
            B = (Xs - Q).norm(2,dim=1).view(1, NX).expand(N, NX)
        if '+' == pm:
            loss = (A-B).clamp(min=0.).mean()
        else:
            loss = (-A+B).clamp(min=0.).mean()
        rank = ((A > B).float().mean()).item()  # normalized result
        return loss, rank
    @staticmethod
    def RankLossCandidateAttackDistance(cs: th.tensor, Qs: th.tensor, Xs: th.tensor, *, metric: str, pm: str):
        '''
        Computes the loss function for pure candidate attack
        using the inferior distance objective
        '''
        assert(cs.shape[1] == Qs.shape[2] == Xs.shape[1])
        NIter, W, D, NX = cs.shape[0], Qs.shape[1], Qs.shape[2], Xs.shape[0]
        losses, ranks = [], []
        for i in range(NIter):
            c = cs[i].view(1, D)
            Q = Qs[i, :, :].view(W, D)
            if (metric, pm) == ('C', '+'):
                loss = (1 - th.mm(c, Q.t())).mean()
            elif (metric, pm) == ('C', '-'):
                loss = -(1 - th.mm(c, Q.t())).mean()
            elif (metric, pm) == ('E', '+'):
                loss = (Q - c).norm(2,dim=1).mean()
            elif (metric, pm) == ('E', '-'):
                loss = -(Q - c).norm(2,dim=1).mean()
            losses.append(loss)
            if metric == 'C':
                A = (1 - th.mm(c, Q.t())).expand(NX, W)
                B = (1 - th.mm(Xs, Q.t()))
            elif metric == 'E':
                A = (Q - c).norm(2, dim=1).expand(NX, W)
                B = (Xs.view(NX,1,D).expand(NX,W,D) - \
                        Q.view(1,W,D).expand(NX,W,D)).norm(2,dim=2)
            rank = ((A > B).float().mean() * NX).item()  # the > sign is correct
            ranks.append(rank)
        loss = th.stack(losses).mean()
        rank = statistics.mean(ranks)
        return (loss, rank)
    def __init__(self, request: str):
        '''
        Initialize various loss functions
        '''
        self.funcmap = {
                'ES': self.RankLossEmbShift,
                'QA': self.RankLossQueryAttack,
                'QA+': functools.partial(self.RankLossQueryAttack, pm='+'),
                'QA-': functools.partial(self.RankLossQueryAttack, pm='-'),
                'QA-DIST': self.RankLossQueryAttackDistance,
                'CA': self.RankLossCandidateAttack,
                'CA+': functools.partial(self.RankLossCandidateAttack, pm='+'),
                'CA-': functools.partial(self.RankLossCandidateAttack, pm='-'),
                'CA-DIST': self.RankLossCandidateAttackDistance,
                'IA-CA': self.RankLossImAgCA,
                'IA-QA': self.RankLossImAgQA,
                }
        if request not in self.funcmap.keys():
            raise KeyError(f'Requested loss function "{request}" not found!')
        self.request = request
    def __call__(self, *args, **kwargs):
        return self.funcmap[self.request](*args, **kwargs)


## MARK: STAGE0
def RankPGD(model, images, labels, candi, *,
        eps=0.3, alpha=1./255., atype=None, M=None, W=None, pm=None,
        verbose=False, device='cpu', loader=None, metric=None,
        normimg=False, transfer=None):
    '''
    Perform FGSM/PGD Query/Candidate attack on the given batch of images, L_infty constraint
    https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks/fast_gradient_method.py

    This is the core of the adversarial ranking implementation,
    but we don't have enough energy to tidy it up before CVPR submission.
    '''
    # >> prepare the current batch of images
    assert(type(images) == th.Tensor)
    images = images.clone().detach().to(device)
    images_orig = images.clone().detach()
    images.requires_grad = True
    labels = labels.to(device).view(-1)
    # >> sanity check for normalized images, if any
    if normimg:
        # normed_img = (image - mean)/std
        IMmean = th.tensor([0.485, 0.456, 0.406], device=device)
        IMstd = th.tensor([0.229, 0.224, 0.225], device=device)
        renorm = lambda im: im.sub(IMmean[:,None,None]).div(IMstd[:,None,None])
        denorm = lambda im: im.mul(IMstd[:,None,None]).add(IMmean[:,None,None])
    if (not normimg) and ((images > 1.0).sum() + (images < 0.0).sum() > 0):
        raise Exception("please toggle 'normimg' as True for sanity")
    def tensorStat(t):
        return f'Min {t.min().item()} Max {t.max().item()} Mean {t.mean().item()}'

    ## special stage : Image agnostic attack
    if int(os.getenv('IAP', 0)) > 0:
        # forward batch
        model.eval()
        with th.no_grad():
            if metric == 'C':
                output = model.forward(images, l2norm=True)
                dist = 1 - output @ candi[0].t()  # [num_output_num, num_candidate]
            elif metric == 'E':
                output = model.forward(images, l2norm=False)
                dist = []
                # the memory requirement is insane if we want to do the pairwise distance
                # matrix in a single step like faC_c2f2_siamese.py's loss function.
                for i in range(output.shape[0]):
                    xq = output[i].view(1, -1)
                    xqd = (candi[0] - xq).norm(2, dim=1).squeeze()
                    dist.append(xqd)
                dist = th.stack(dist)  # [num_output_num, num_candidate]
            else:
                raise ValueError(metric)
            output_orig = output.clone().detach()
        # select attacking target and start
        lrankiap, lrankiapb = [], [] # do return
        lrankiapt = []  # no report
        lorankiap, lorankiapb = [], []  # do return
        for iapiter in range(int(output.size(0)/10)):
            random_select = random.randint(0, candi[0].size(0)-1)
            iap_target = candi[0][random_select].view(1, -1)
            if metric == 'C':
                # refdist [N] = iap_target [1, D] * output_orig.t() [D, N]
                refdist = (1 - th.mm(iap_target, output_orig.t())).view(-1)
            else:
                # refdist [N] = (output_orig [N,D] - iap_target [1,D]) |> norm2
                refdist = (output_orig - iap_target).norm(2,dim=1).view(-1)
            # iap_target = output[0].view(1, -1) # for debugging
            # try to generate image agnostic perturbation
            if False:
                # perturbation on a per-sample basis
                iap = th.zeros(images.shape, device=images.device, requires_grad=True)
            else:
                # universal perturbation
                iap = th.zeros((1, *(images.shape[1:])),
                        device=images.device, requires_grad=True)
            optimM_iap = th.optim.SGD(model.parameters(), lr=1.)
            for iteration in range(100):
                optim_iap = th.optim.SGD([iap], lr=1.)
                optimM_iap.zero_grad(); optim_iap.zero_grad()
                # first half for white-box
                output = model.forward(images_orig[0::2] + iap,
                        l2norm=(True if metric == 'C' else False))
                if ('ES' == atype):
                    raise NotImplementedError
                elif (atype in ('FOA', 'SPFOA')):
                    raise NotImplementedError
                elif ('CA' == atype):
                    loss_iap, rank_iap = LossFactory('IA-CA')(
                            output, iap_target, candi[0], metric=metric, pm=pm)
                    #print(_fgcGrey(f'(IAP-it{iteration})', 'loss=', loss_iap.item(),
                    #        f'CA{pm}:rank=', rank_iap))
                elif ('QA' == atype):
                    loss_iap, rank_iap = LossFactory('IA-QA')(
                            output, iap_target, candi[0], metric=metric, pm=pm)
                    #print(_fgcGrey(f'(IAP-it{iteration})', 'loss=', loss_iap.item(),
                    #        f'QA{pm}:rank=', rank_iap))
                elif ('SPQA' == atype):
                    raise NotImplementedError
                else:
                    raise Exception("unknown IAP attack")
                loss_iap.backward()
                if normimg:
                    raise NotImplementedError
                else:
                    iap.grad.data.copy_(2./255. * th.sign(iap.grad))
                    optim_iap.step() # gradient descent
                    # projection
                    iap = th.min(iap, 1 - images_orig)
                    iap = th.max(iap, 0 - images_orig)
                    iap = th.min(iap, th.tensor(eps, device=iap.device))
                    iap = th.max(iap, th.tensor(-eps, device=iap.device))
                    iap = iap.detach().mean(0, keepdim=True)
                    iap.requires_grad = True
            assert(iap.size(0) == 1)  # singal universal perturbation
            with th.no_grad():
                output = model.forward((images_orig + iap).clamp(min=0.,max=1.),
                        l2norm=(True if metric == 'C' else False))
                if ('ES' == atype):
                    raise NotImplementedError
                elif ('FOA' == atype) or ('SPFOA' == atype):
                    raise NotImplementedError
                elif ('CA' == atype):
                    _, rank_iap = LossFactory('IA-CA')(
                            output[0::2], iap_target, candi[0], metric=metric, pm=pm)
                    print(_fgcGrey(f'(IAP-white)', 'loss=%.5f'% loss_iap.item(),
                            f'CA{pm}:rank= %.4f'%rank_iap,
                            'min=%.3f'%iap.min().item(),
                            'max=%.3f'%iap.max().item(), 'mean=%.3f'%iap.mean().item()))
                    _, rank_iapb = LossFactory('IA-CA')(
                            output[1::2], iap_target, candi[0], metric=metric, pm=pm)
                    print(_fgcGrey(f'(IAP-black)', f'CA{pm}:rank= %.3f'%rank_iapb))
                    # what about topK ?
                    if '-' == pm:
                        _topk_ = max(1, int(0.01 * output_orig.size(0)/2)) # 1%
                        output_w_tk = output[0::2][refdist[0::2].topk(_topk_, largest=False)[1]]
                        output_b_tk = output[1::2][refdist[1::2].topk(_topk_, largest=False)[1]]
                        orig_output_w_tk = output_orig[0::2][refdist[0::2].topk(_topk_, largest=False)[1]]
                        orig_output_b_tk = output_orig[1::2][refdist[1::2].topk(_topk_, largest=False)[1]]
                        _, rankiapwtk = LossFactory('IA-CA')(
                                output_w_tk, iap_target, candi[0], metric=metric, pm=pm)
                        _, rankiapbtk = LossFactory('IA-CA')(
                                output_b_tk, iap_target, candi[0], metric=metric, pm=pm)
                        _, orankiapwtk = LossFactory('IA-CA')(
                                orig_output_w_tk, iap_target, candi[0], metric=metric, pm=pm)
                        _, orankiapbtk = LossFactory('IA-CA')(
                                orig_output_b_tk, iap_target, candi[0], metric=metric, pm=pm)
                        print('<IAP- Topk>', 'white:rank= %.3f -> %.3f'%(orankiapwtk, rankiapwtk),
                                'black:rank= %.3f -> %.3f'%(orankiapbtk, rankiapbtk))
                        rank_iap = rankiapwtk
                        rank_iapb = rankiapbtk
                        lorankiap.append(orankiapwtk)
                        lorankiapb.append(orankiapbtk)
                    else:
                        lorankiap.append(0)
                        lorankiapb.append(0)
                elif ('QA' == atype):
                    _, rank_iap = LossFactory('IA-QA')(
                            output[0::2], iap_target, candi[0], metric=metric, pm=pm)
                    print(_fgcGrey(f'(IAP-white)', 'loss=%.5f'% loss_iap.item(),
                            f'QA{pm}:rank= %.4f'%rank_iap,
                            'min=%.3f'%iap.min().item(),
                            'max=%.3f'%iap.max().item(), 'mean=%.3f'%iap.mean().item()))
                    _, rank_iapb = LossFactory('IA-QA')(
                            output[1::2], iap_target, candi[0], metric=metric, pm=pm)
                    print(_fgcGrey(f'(IAP-black)', f'QA{pm}:rank= %.3f'%rank_iapb))
                    # what about topK ?
                    if '-' == pm:
                        _topk_ = max(1, int(0.01 * output_orig.size(0)/2)) # 2%
                        output_w_tk = output[0::2][refdist[0::2].topk(_topk_, largest=False)[1]]
                        output_b_tk = output[1::2][refdist[1::2].topk(_topk_, largest=False)[1]]
                        orig_output_w_tk = output_orig[0::2][refdist[0::2].topk(_topk_, largest=False)[1]]
                        orig_output_b_tk = output_orig[1::2][refdist[1::2].topk(_topk_, largest=False)[1]]
                        _, rankiapwtk = LossFactory('IA-QA')(
                                output_w_tk, iap_target, candi[0], metric=metric, pm=pm)
                        _, rankiapbtk = LossFactory('IA-QA')(
                                output_b_tk, iap_target, candi[0], metric=metric, pm=pm)
                        _, orankiapwtk = LossFactory('IA-QA')(
                                orig_output_w_tk, iap_target, candi[0], metric=metric, pm=pm)
                        _, orankiapbtk = LossFactory('IA-QA')(
                                orig_output_b_tk, iap_target, candi[0], metric=metric, pm=pm)
                        print('<IAP- Topk>', 'white:rank= %.3f -> %.3f'%(orankiapwtk, rankiapwtk),
                                'black:rank= %.3f -> %.3f'%(orankiapbtk, rankiapbtk))
                        rank_iap = rankiapwtk
                        rank_iapb = rankiapbtk
                        lorankiap.append(orankiapwtk)
                        lorankiapb.append(orankiapbtk)
                    else:
                        lorankiap.append(0)
                        lorankiapb.append(0)
            lrankiap.append(rank_iap)
            lrankiapb.append(rank_iapb)
            # transfer attack
            if transfer is not None:
                if 'C' in transfer['transfer']:
                    output_trans = transfer['model'].forward((images+iap).clamp(min=0.,max=1.),
                            l2norm=(True if metric == 'C' else False))
                    iap_target_trans = transfer['candidates'][0][random_select].view(1, -1)
                    if ('CA' == atype):
                        _, rank_iapt = LossFactory('IA-CA')(
                                output_trans, iap_target_trans, transfer['candidates'][0],
                                metric=metric, pm=pm)
                        print('<IAP-transfer>', f'CA{pm}:rank=', rank_iapt)
                        lrankiapt.append(rank_iapt)
                    elif ('QA' == atype):
                        _, rank_iapt = LossFactory('IA-QA')(
                                output_trans, iap_target_trans, transfer['candidates'][0],
                                metric=metric, pm=pm)
                        print('<IAP-transfer>', f'QA{pm}:rank=', rank_iapt)
                        lrankiapt.append(rank_iapt)
                    else:
                        raise NotImplementedError
        # historical burden
        print('(IAP-Batch)', atype, metric, pm, statistics.mean(lrankiap),
                statistics.mean(lrankiapb))
        if transfer is not None:
            print('<IAP-transfer>', 'rank=', statistics.mean(lrankiapt))
        return None, None, (output_orig, output), \
                (0, 0), (
                        [0, 0],
                        [0, statistics.mean(lrankiap), statistics.mean(lrankiapb),
                            statistics.mean(lorankiap), statistics.mean(lorankiapb)]
                        )

    # END special stage

    #<<<<<< STAGE1: ORIG SAMPLE EVALUATION <<<<<<
    model.eval()
    with th.no_grad():

        # -- [orig] -- forward the original samples with the original loss
        # >> Result[output]: embedding vectors
        # >> Result[dist]: distance matrix (current batch x database)
        if metric == 'C':
            output = model.forward(images, l2norm=True)
            dist = 1 - output @ candi[0].t()  # [num_output_num, num_candidate]
        elif metric == 'E':
            output = model.forward(images, l2norm=False)
            dist = []
            # the memory requirement is insane if we want to do the pairwise distance
            # matrix in a single step like faC_c2f2_siamese.py's loss function.
            for i in range(output.shape[0]):
                xq = output[i].view(1, -1)
                xqd = (candi[0] - xq).norm(2, dim=1).squeeze()
                dist.append(xqd)
            dist = th.stack(dist)  # [num_output_num, num_candidate]
        else:
            raise ValueError(metric)
        output_orig = output.clone().detach()
        dist_orig = dist.clone().detach()
        loss = th.tensor(-1) # we don't track this value anymore
        loss_orig = th.tensor(-1) # we don't track this value anymore

        #== <transfer> forward the samples with the transfer model
        if transfer is not None:
            if 'C' in transfer['transfer']:
                output_trans = transfer['model'].forward(images, l2norm=True)
                dist_trans = 1 - output_trans @ transfer['candidates'][0].t()
            elif 'E' in transfer['transfer']:
                output_trans = transfer['model'].forward(images, l2norm=False)
                dist_trans = []
                for i in range(output_trans.shape[0]):
                    xtrans = output_trans[i].view(1, -1)
                    xdtrans = (transfer['candidates'][0] - xtrans).norm(2, dim=1).squeeze()
                    dist_trans.append(xdtrans)
                dist_trans = th.stack(dist_trans)

        # -- [orig] -- select attack targets and calculate the attacking loss
        if ('ES' == atype) and (M is None) and (W is None):
            # -- [orig] untargeted attack, UT
            # >> the KNN accuracy, or say the Recall@1
            allLab = candi[1].cpu().numpy().squeeze()
            localLab = labels.cpu().numpy().squeeze()
            r_1, r_10, r_100 = [], [], []
            for i in range(dist.shape[0]):
                agsort = dist[i].cpu().numpy().argsort()[1:]
                rank = np.where(allLab[agsort] == localLab[i])[0].min()
                r_1.append(rank == 0)
                r_10.append(rank < 10)
                r_100.append(rank < 100)
            r_1, r_10, r_100 = 100*np.mean(r_1), 100*np.mean(r_10), 100*np.mean(r_100)

            # >> backup and report
            correct_orig = r_1*dist.shape[0]/100 #knnclass.eq(labels.cpu()).sum().item()
            knnsearch_orig = dist.sort(dim=1)[1].flatten() # for visulization
            if verbose:
                print()
                print('* Original Sample', 'r@1=', r_1, 'r@10=', r_10, 'r@100=', r_100)

            # <transfer>
            if transfer is not None:
                alllabel_trans = transfer['candidates'][1].cpu().numpy().squeeze()
                r_1, r_10, r_100 = [], [], []
                for i in range(dist_trans.shape[0]):
                    agsort = dist_trans[i].cpu().numpy().argsort()[1:]
                    rank = np.where(alllabel_trans[agsort] == localLab[i])[0].min()
                    r_1.append(rank == 0)
                    r_10.append(rank < 10)
                    r_100.append(rank < 100)
                r_1, r_10, r_100 = 100*np.mean(r_1), 100*np.mean(r_10), 100*np.mean(r_100)
                print('* <transfer> Original Sample', 'r@1=', r_1, 'r@10=', r_10, 'r@100=', r_100)

        elif ('CA' == atype) and (W is not None):
            # -- [orig] candidate attack, W=?
            # >> select W=? attacking targets
            if '+' == pm:
                if 'global' == os.getenv('SAMPLE', 'global'):
                    msample = th.randint(candi[0].shape[0], (output.shape[0], W)) # [output_0, W]
                    #msample[:,0] = 14649
                    #msample[:,1] = 11000
                    #msample[:,2] = 14706
                    #msample[:,3] = 14703
                    #msample[:,4] = 14742
                    #msample[:,5] = 14782
                    #msample[:,6] = 14801
                    #msample[:,7] = 14890
                    #msample[:,8] = 14903
                    #msample[:,9] = 14930
                    # MARKX
                elif 'local' == os.getenv('SAMPLE', 'global'):
                    local_lb = int(candi[0].shape[0]*0.01)
                    local_ub = int(candi[0].shape[0]*0.05)
                    topxm = dist.topk(local_ub+1, dim=1, largest=False)[1][:,1:]
                    sel = np.random.randint(local_lb, local_ub, (output.shape[0], W))
                    msample = th.stack([topxm[i][sel[i]] for i in range(topxm.shape[0])])
            elif '-' == pm:
                # these are not the extremely precise topW queries but an approximation
                # select W candidates from the topmost samples
                topmost = int(candi[0].size(0) * 0.01)
                if int(os.getenv('VIS', 0)) > 0:
                    topmost = int(candi[0].size(0) * 0.0003)
                if transfer is None:
                    topxm = dist.topk(topmost+1, dim=1, largest=False)[1][:,1:]  # [output_0, W]
                else:
                    topxm = dist_trans.topk(topmost+1, dim=1, largest=False)[1][:,1:]
                sel = np.random.randint(0, topmost, [topxm.shape[0], W])
                msample = th.stack([topxm[i][sel[i]] for i in range(topxm.shape[0])])
            embpairs = candi[0][msample, :] # [output_0, W, output_1]

            #== evaluate the original sample
            loss, rank = LossFactory(f'CA{pm}')(output, embpairs, candi[0], metric=metric)
            mrank = rank / candi[0].shape[0]

            # >> backup and report
            correct_orig = mrank * output.shape[0] / 100.
            loss_orig = loss.clone().detach()
            if verbose:
                print()
                print('* Original Sample', 'loss=', loss.item(), f'CA{pm}:rank=', mrank)

            # <transfer>
            if transfer is not None:
                embpairs_trans = transfer['candidates'][0][msample, :]
                _, rank_trans = LossFactory('CA')(output_trans, embpairs_trans,
                        transfer['candidates'][0], pm=pm,
                        metric=('C' if 'C' in transfer['transfer'] else 'E'))
                print('* <transfer> Original Sample',
                        f'CA{pm}:rank=', rank_trans / candi[0].shape[0])

        elif (atype in ['QA', 'SPQA']) and (M is not None):
            #== semantic-preserving (SP) query attack
            #; the pure query attack has a downside: its semantic may be changed
            #; during the attack. That would result in a very weird ranking result,
            #; even if the attacking goal was achieved.

            #== configuration
            M_GT = 5 # sensible due to the SOP dataset property
            XI = float(os.getenv('SP', 1. if ('+' == pm) else 100.)) # balancing the "SP" and "QA" loss functions
            if 'SP' not in atype:
                XI = None

            #== first, select the attacking targets and the ground-truth vectors for
            #; the SP purpose.
            if '+' == pm:
                # random sampling from populationfor QA+
                if 'global' == os.getenv('SAMPLE', 'global'):
                    msample = th.randint(candi[0].shape[0], (output.shape[0], M)) # [output_0,M]
                elif 'local' == os.getenv('SAMPLE', 'global'):
                    local_lb = int(candi[0].shape[0]*0.01)
                    local_ub = int(candi[0].shape[0]*0.05)
                    topxm = dist.topk(local_ub+1, dim=1, largest=False)[1][:,1:]
                    sel = np.random.randint(local_lb, local_ub, (output.shape[0], M))
                    msample = th.stack([topxm[i][sel[i]] for i in range(topxm.shape[0])])
                if 'SP' in atype:
                    mgtruth = dist.topk(M_GT+1, dim=1, largest=False)[1][:,1:] # [output_0, M]
            elif '-' == pm:
                # random sampling from top-3M for QA-
                topmost = int(candi[0].size(0) * 0.01)
                if int(os.getenv('VIS', 0)) > 0:
                    topmost = int(candi[0].size(0) * 0.0003)
                if transfer is None:
                    topxm = dist.topk(topmost+1, dim=1, largest=False)[1][:,1:]
                else:
                    topxm = dist_trans.topk(topmost+1, dim=1, largest=False)[1][:,1:]
                sel = np.vstack([np.random.permutation(topmost) for i in range(output.shape[0])])
                msample = th.stack([topxm[i][sel[i,:M]] for i in range(topxm.shape[0])])
                if 'SP' in atype:
                    mgtruth = th.stack([topxm[i][np.sort(sel[i,M:])[:M_GT]] for i in range(topxm.shape[0])])
            embpairs = candi[0][msample, :]
            if 'SP' in atype:
                embgts = candi[0][mgtruth, :]

            #== evaluate the SPQA loss on original samples
            if 'SP' in atype:
                loss_qa, rank_qa = LossFactory('QA')(output, embpairs, candi[0],
                        metric=metric, pm=pm, dist=dist, cidx=msample)
                loss_sp, rank_sp = LossFactory('QA+')(output, embgts, candi[0],
                        metric=metric, dist=dist, cidx=mgtruth)
                loss = loss_qa + XI * loss_sp
            else:
                loss_qa, rank_qa = LossFactory('QA')(output, embpairs, candi[0],
                        metric=metric, pm=pm, dist=dist, cidx=msample)
                loss = loss_qa

            #== overall loss function of the batch
            mrank = rank_qa / candi[0].shape[0]
            correct_orig = mrank * output.shape[0] / 100.
            loss_orig = loss.clone().detach()
            if 'SP' in atype:
                mrankgt = rank_sp / candi[0].shape[0]
                sp_orig = mrankgt * output.shape[0] / 100.
                prankgt_orig = mrankgt
            #== backup and report
            if verbose:
                print()
                if 'SP' in atype:
                    print('* Original Sample', 'loss=', loss.item(),
                            f'SPQA{pm}:rank=', mrank,
                            f'SPQA{pm}:GTrank=', mrankgt)
                else:
                    print('* Original Sample', 'loss=', loss.item(),
                            f'QA{pm}:rank=', mrank)

            # <transfer>
            if transfer is not None:
                embpairs_trans = transfer['candidates'][0][msample, :]
                _, rank_qa_trans = LossFactory('QA')(output_trans, embpairs_trans,
                        transfer['candidates'][0], pm=pm,
                        metric=('C' if 'C' in transfer['transfer'] else 'E'),
                        dist=dist_trans, cidx=msample)
                if 'SP' in atype:
                    embgts_trans = transfer['candidates'][0][mgtruth, :]
                    _, rank_sp_trans = LossFactory('QA')(output_trans, embgts_trans,
                            transfer['candidates'][0], pm=pm,
                            metric=('C' if 'C' in transfer['transfer'] else 'E'),
                            dist=dist_trans, cidx=mgtruth)
                if 'SP' not in atype:
                    print('* <transfer> Original Sample', f'QA{pm}:rank=', rank_qa_trans / candi[0].shape[0])
                else:
                    print('* <transfer> Original Sample', f'SPQA{pm}:rank=', rank_qa_trans / candi[0].shape[0],
                            f'SPQA{pm}:GTrank=', rank_sp_trans / candi[0].shape[0])
        else:
            raise Exception("Unknown attack")
    # >>>>>>>>>>>>>>>>>>>>>>>>>> ORIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # <<<<<<<<<<<<<<<<<<<<<<<<<< MARK: STATTACK <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # -- [attack] start attack, maxiter is the number of PGD iterations (FGSM: maxiter=1)

    # -- [alpha/epsilon] PGD parameter tricks balancing yield v.s. time consumption
    #    Note, 1/255 \approx 0.004 
    if eps > 1./255.:
        alpha = float(min(max(eps/10., 1./255.), 0.01))  # PGD hyper-parameter
        maxiter = int(min(max(10, 2*eps/alpha), 30))  # PGD hyper-parameter
    elif eps < 1./255.:
        maxiter = 1
        alpha = eps / max(1, maxiter//2)  # doesn't attack

    # this ensures the best attacking result, but is very time-consuming
    if int(os.getenv('PGD_UTMOST', 0)) > 0:
        print('| Overriding PGD Parameter for the utmost attacking result|')
        alpha = 1./255.
        maxiter = int(os.getenv('PGD_UTMOST'))

    # PGD with/without random init?
    if int(os.getenv('RINIT', 0))>0:
        if not normimg:
            images = images + eps*2*(0.5-th.rand(images.shape)).to(images.device)
            images = th.clamp(images, min=0., max=1.)
            images = images.detach()
            images.requires_grad = True
        else:
            images = images + (eps/IMstd[:,None,None])*2*(0.5-th.rand(images.shape)).to(images.device)
            images = th.max(images, renorm(th.zeros(images.shape).to(device)))
            images = th.min(images, renorm(th.ones(images.shape).to(device)))
            images = images.detach()
            images.requires_grad = True

    # truning the model into tranining mode triggers obscure problem:
    # incorrect validation performance due to BatchNorm. We do automatic
    # differentiation in evaluation mode instead.
    #model.train()
    for iteration in range(maxiter):

        # >> prepare optimizer for SGD
        optim = th.optim.SGD(model.parameters(), lr=1.)
        optimx = th.optim.SGD([images], lr=1.)
        optim.zero_grad(); optimx.zero_grad()
        output = model.forward(images, l2norm=(True if metric=='C' else False))

        if ('ES' == atype) and (M is None) and (W is None): # -- [attack] untargeted, UT
            # >> push the embedding off the original location, the further the better.
            if metric == 'C':
                distance = 1 - th.mm(output, output_orig.t())
                loss = -distance.trace() # gradient ascent on trace, i.e. diag.sum
            elif metric == 'E':
                distance = th.nn.functional.pairwise_distance(output, output_orig, p=2)
                loss = -distance.sum()
            if verbose:
                print(_fgcGrey('(PGD) iter', iteration, 'loss', loss.item()))
        elif ('CA' == atype) and (W is not None): # -- [attack] candidate attack, W=?
            # >> enforce the target set of inequalities
            if int(os.getenv('DISTANCE', 0)) > 0:
                loss, _ = LossFactory('CA-DIST')(output, embpairs, candi[0], metric=metric, pm=pm)
            else:
                loss, _ = LossFactory('CA')(output, embpairs, candi[0], metric=metric, pm=pm)
            if verbose:
                print(_fgcGrey('(PGD) iter', iteration, 'loss', loss.item()))
        elif (atype in ['QA', 'SPQA']) and (M is not None):
            #== enforce the target set of inequalities, while preserving the semantic
            if int(os.getenv('DISTANCE', 0)) > 0:
                if 'SP' in atype:
                    #raise NotImplementedError("SP for distance based objective makes no sense here")
                    loss_qa, _ = LossFactory('QA-DIST')(output, embpairs, candi[0], metric=metric, pm=pm)
                    loss_sp, _ = LossFactory('QA-DIST')(output, embgts, candi[0], metric=metric, pm='+')
                    loss = loss_qa + XI * loss_sp
                    if verbose:
                        print(_fgcGrey('(PGD) iter', iteration, 'loss', loss.item()))
                else:
                    loss, _ = LossFactory('QA-DIST')(output, embpairs, candi[0], metric=metric, pm=pm)
                    if verbose:
                        print(_fgcGrey('(PGD) iter', iteration, 'loss', loss.item()))
            elif 'SP' in atype:
                loss_qa, _ = LossFactory('QA')(output, embpairs, candi[0], metric=metric, pm=pm)
                loss_sp, _ = LossFactory('QA')(output, embgts, candi[0], metric=metric, pm='+')
                loss = loss_qa + XI * loss_sp
                if verbose:
                    print(_fgcGrey('(PGD) iter', iteration, 'loss', loss.item(),
                        '\t|optim|', f'loss:QA{pm}=', loss_qa.item()
                        , 'loss:SP(QA+)=', loss_sp.item()))
            else:
                loss, _ = LossFactory('QA')(output, embpairs, candi[0], metric=metric, pm=pm)
                if verbose:
                    print(_fgcGrey('(PGD) iter', iteration, 'loss', loss.item()))
        else:
            raise Exception("Unknown attack")
        # >> Do FGSM/PGD update step
        # >> PGD projection when image has been normalized is a little bit special
        loss.backward()
        if not normimg:
            if maxiter>1:
                images.grad.data.copy_(alpha * th.sign(images.grad))
            elif maxiter==1:
                images.grad.data.copy_(eps * th.sign(images.grad))  # FGSM
            # >> PGD: project SGD optimized result back to a valid region
            optimx.step()
            images = th.min(images, images_orig + eps) # L_infty constraint
            images = th.max(images, images_orig - eps) # L_infty constraint
            images = th.clamp(images, min=0., max=1.)
            images = images.clone().detach()
            images.requires_grad = True
        else: #normimg:
            if maxiter>1:
                images.grad.data.copy_((alpha/IMstd[:,None,None])*th.sign(images.grad))
            elif maxiter==1:
                images.grad.data.copy_((eps/IMstd[:,None,None])*th.sign(images.grad))
            else:
                raise Exception("Your arguments must be wrong")
            # >> PGD: project SGD optimized result back to a valid region
            optimx.step()
            images = th.min(images, images_orig + (eps/IMstd[:,None,None]))
            images = th.max(images, images_orig - (eps/IMstd[:,None,None]))
            #images = th.clamp(images, min=0., max=1.)
            images = th.max(images, renorm(th.zeros(images.shape).to(device)))
            images = th.min(images, renorm(th.ones(images.shape).to(device)))
            images = images.clone().detach()
            images.requires_grad = True

    # >>>>>>>>>>>>>>>>>>>>>>>> END ATTACK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # <<<<<<<<<<<<<<<<<<<<<<<< ADVERSARIAL <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # --- [eval adv] re-evaluate these off-manifold adversarial samples
    xr = images
    r = images - images_orig
    model.eval()
    with th.no_grad():
        # >> forward the adversarial examples, and optionally calculate loss
        if metric == 'C':
            output = model.forward(images, l2norm=True)
            dist = 1 - output @ candi[0].t()
            loss = th.tensor(-1)
        elif metric == 'E':
            output = model.forward(images, l2norm=False)
            dist = []
            # the memory requirement is insane if we want to do the pairwise distance
            # matrix in a single step like faC_c2f2_siamese.py's loss function.
            for i in range(output.shape[0]):
                xq = output[i].view(1, -1)
                xqd = (candi[0] - xq).norm(dim=1).squeeze()
                dist.append(xqd)
            dist = th.stack(dist)
            loss = th.tensor(-1)
        else:
            raise ValueError(metric)
        output_adv = output.clone().detach()
        dist_adv = dist.clone().detach()

        # <transfer>
        if transfer is not None:
            if 'C' in transfer['transfer']:
                output_trans = transfer['model'].forward(images, l2norm=True)
                dist_trans = 1 - output_trans @ transfer['candidates'][0].t()
            elif 'E' in transfer['transfer']:
                output_trans = transfer['model'].forward(images, l2norm=False)
                dist_trans = []
                for i in range(output_trans.shape[0]):
                    xtrans = output_trans[i].view(1, -1)
                    xdtrans = (transfer['candidates'][0] - xtrans).norm(2, dim=1).squeeze()
                    dist_trans.append(xdtrans)
                dist_trans = th.stack(dist_trans)

        # >> calculate embedding shift
        if metric == 'C':
            distance = 1 - th.mm(output, output_orig.t())
            embshift = distance.trace()/output.shape[0] # i.e. trace = diag.sum
            embshift = embshift.item()
        elif metric == 'E':
            distance = th.nn.functional.pairwise_distance(output, output_orig, p=2)
            embshift = distance.sum()/output.shape[0]
            embshift = embshift.item()

        if ('ES' == atype) and (M is None) and (W is None):
            # -- [eval adv] untargeted, UT
            allLab = candi[1].cpu().numpy()
            localLab = labels.cpu().numpy()
            r_1, r_10, r_100 = [], [], []
            for i in range(dist.shape[0]):
                if metric == 'C':
                    loc = (1 - candi[0] @ output_orig[i].view(-1,1)).flatten().argmin().cpu().numpy()
                else:
                    loc = (candi[0] - output_orig[i]).norm(2,dim=1).flatten().argmin().cpu().numpy()
                dist[i][loc] = 1e30
                agsort = dist[i].cpu().numpy().argsort()[0:]
                rank = np.where(allLab[agsort] == localLab[i])[0].min()
                r_1.append(rank == 0)
                r_10.append(rank < 10)
                r_100.append(rank < 100)
            r_1, r_10, r_100 = 100*np.mean(r_1), 100*np.mean(r_10), 100*np.mean(r_100)

            correct_adv = r_1*dist.shape[0]/100
            output_adv = output.clone().detach()
            loss_adv = loss.clone().detach()

            rankup = 0
            if verbose:
                print('* Adversar Sample', 'r@1=', r_1, 'r@10=', r_10, 'r@100=', r_100,
                        'embShift=', embshift)

            if transfer is not None:
                r_1, r_10, r_100 = [], [], []
                for i in range(dist_trans.shape[0]):
                    if metric == 'C':
                        loc = (1 - candi[0] @ output_orig[i].view(-1,1)).flatten().argmin().cpu().numpy()
                    else:
                        loc = (candi[0] - output_orig[i]).norm(2,dim=1).flatten().argmin().cpu().numpy()
                    dist_trans[i][loc] = 1e30
                    agsort = dist_trans[i].cpu().numpy().argsort()[0:]
                    rank = np.where(alllabel_trans[agsort] == localLab[i])[0].min()
                    r_1.append(rank == 0)
                    r_10.append(rank < 10)
                    r_100.append(rank < 100)
                r_1, r_10, r_100 = 100*np.mean(r_1), 100*np.mean(r_10), 100*np.mean(r_100)
                print('* <transfer> Adversarial Sample', 'r@1=', r_1, 'r@10=', r_10, 'r@100=', r_100)
        elif ('CA' == atype) and (W is not None):
            # -- [eval adv] candidate attack, W=?
            loss, rank = LossFactory('CA')(output, embpairs, candi[0], metric=metric, pm=pm)
            mrank = rank / candi[0].shape[0]
            correct_adv = mrank * output.shape[0] / 100.
            loss_adv = loss.clone().detach()
            rankup = (correct_adv - correct_orig) * 100. / output.shape[0]
            if verbose:
                print('* Adversarial example', 'loss=', loss.item(),
                        f'CA{pm}:rank=', mrank, 'embShift=', embshift)
            if transfer is not None:
                _, rank_trans = LossFactory('CA')(output_trans, embpairs_trans,
                        transfer['candidates'][0], pm=pm,
                        metric=('C' if 'C' in transfer['transfer'] else 'E'))
                prank_trans = rank_trans / candi[0].size(0)
                print('* <transfer> Adversarial Sample', f'CA{pm}:rank=', prank_trans)
        elif (atype in ['QA', 'SPQA']) and (M is not None):
            # evaluate the adversarial examples
            if 'SP' in atype:
                loss_qa, rank_qa = LossFactory('QA')(output, embpairs, candi[0],
                        metric=metric, pm=pm, dist=dist, cidx=msample)
                loss_sp, rank_sp = LossFactory('QA')(output, embgts, candi[0],
                        metric=metric, pm='+', dist=dist, cidx=mgtruth)
                loss = loss_qa + XI * loss_sp
            else:
                loss_qa, rank_qa = LossFactory('QA')(output, embpairs, candi[0],
                        metric=metric, pm=pm, dist=dist, cidx=msample)
                loss = loss_qa
            mrank = rank_qa / candi[0].shape[0]
            correct_adv = mrank * output.shape[0] / 100.
            loss_adv = loss.clone().detach()
            rankup = (correct_adv - correct_orig) * 100. / output.shape[0]
            if 'SP' in atype:
                mrankgt = rank_sp / candi[0].shape[0]
                sp_adv = mrankgt * output.shape[0] / 100.
                prankgt_adv = mrankgt
            if verbose:
                if 'SP' in atype:
                    print('* Adversarial Sample', 'loss=', loss.item(),
                            f'SPQA{pm}:rank=', mrank,
                            f'SPQA{pm}:GTrank=', mrankgt,
                            'embShift=', embshift)
                else:
                    print('* Adversarial Sample', 'loss=', loss.item(),
                            f'QA{pm}:rank=', mrank, 'embShift=', embshift)
            # <transfer>
            if transfer is not None:
                _, rank_qa_trans = LossFactory('QA')(output_trans, embpairs_trans,
                        transfer['candidates'][0], pm=pm,
                        metric=('C' if 'C' in transfer['transfer'] else 'E'),
                        dist=dist_trans, cidx=msample)
                if 'SP' in atype:
                    _, rank_sp_trans = LossFactory('QA')(output_trans, embgts_trans,
                            transfer['candidates'][0], pm=pm,
                            metric=('C' if 'C' in transfer['transfer'] else 'E'),
                            dist=dist_trans, cidx=mgtruth)
                if 'SP' not in atype:
                    print('* <transfer> Adversarial Sample',
                            f'QA{pm}:rank=', rank_qa_trans / candi[0].shape[0])
                else:
                    prank_trans = rank_qa_trans / candi[0].size(0)
                    print('* <transfer> Adversarial Sample',
                            f'SPQA{pm}:rank=', prank_trans,
                            f'SPQA{pm}:GTrank=', rank_sp_trans / candi[0].shape[0])
        else:
            raise Exception("Unknown attack")
    prankgt_orig = locals().get('prankgt_orig', -1.)
    prankgt_adv = locals().get('prankgt_adv', -1.)
    prank_trans = locals().get('prank_trans', -1.)
    # >>>>>>>>>>>>>>>>>>>>>> ADVERSARIAL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # -- [report] statistics and optionally visualize
    if verbose:
        if normimg:
            r = r.mul(IMstd[:,None,None])
        print(_fgcGrey('r>', 'Min', '%.3f'%r.min().item(),
                'Max', '%.3f'%r.max().item(),
                'Mean', '%.3f'%r.mean().item(),
                'L0', '%.3f'%r.norm(0).item(),
                'L1', '%.3f'%r.norm(1).item(),
                'L2', '%.3f'%r.norm(2).item()))

    # [[trigger:env]] visualization:entrance
    if int(os.getenv('VIS', 0)) > 0:
        # visualize some examples
        if 'CA' == atype and '+' == pm:
            if len(msample[0]) == 1:
                VisualizorFactory('CA+')(xr, r, images_orig, output_orig,
                        output_adv, msample, loader, candi[0],
                        metric=metric, normimg=normimg)
            elif len(msample[0]) == 2:
                VisualizorFactory('CA+2')(xr, r, images_orig, output_orig,
                        output_adv, msample, loader, candi[0],
                        metric=metric, normimg=normimg)
            elif len(msample[0]) == 10:
                VisualizorFactory('CA+10')(xr, r, images_orig, output_orig,
                        output_adv, msample, loader, candi[0],
                        metric=metric, normimg=normimg)
            else:
                raise NotImplementedError
        elif 'CA' == atype and '-' == pm:
            VisualizorFactory('CA-')(xr, r, images_orig, output_orig,
                    output_adv, msample, loader, candi[0],
                    metric=metric, normimg=normimg)
        elif atype in ('QA', 'SPQA') and '+' == pm:
            if len(msample[0]) == 1:
                VisualizorFactory('QA+')(xr, r, images_orig, output_orig,
                        output_adv, msample, loader, candi[0],
                        metric=metric, normimg=normimg)
            elif len(msample[0]) == 2:
                VisualizorFactory('QA+2')(xr, r, images_orig, output_orig,
                        output_adv, msample, loader, candi[0],
                        metric=metric, normimg=normimg)
        elif atype in ('QA', 'SPQA') and '-' == pm:
            VisualizorFactory('QA-')(xr, r, images_orig, output_orig,
                    output_adv, msample, loader, candi[0],
                    metric=metric, normimg=normimg)
        else:
            raise NotImplementedError(atype)

    # historical burden, return data/stat of the current batch
    return xr, r, (output_orig, output), \
            (loss_orig, loss), (
                    [correct_orig, None],
                    [correct_adv, rankup, embshift, prankgt_adv, prank_trans]
                    )

class VisualizorFactory(object):
    '''
    Collection of Attack Visualization Routines
    '''
    def __init__(self, typ):
        self._factory = {
            'ES': VisualizorFactory.vis_es,
            'CA+': VisualizorFactory.vis_caplus,
            'CA+2': VisualizorFactory.vis_caplus2,
            'CA+10': VisualizorFactory.vis_caplus10,
            'CA-': VisualizorFactory.vis_caminus,
            'QA+': VisualizorFactory.vis_qaplus,
            'QA+2': VisualizorFactory.vis_qaplus2,
            'QA-': VisualizorFactory.vis_qaminus,
        }
        self.typ = typ
    def __call__(self, *args, **kwargs):
        return self._factory[self.typ](*args, **kwargs)

    @staticmethod
    def vis_caplus(xr, r, imorig, # perturbed images, perturb, original images
            outorig, outadv, # original/adversarial distance matrix
            targetidx, # Q:idx for CA, C:idx for QA
            loader, # for loading images outside of the batch
            cddts,
            *,
            normimg=False, # for SOP
            metric=None,
            ):
        '''
        visualize CA+ effects
        '''
        if len(r[0].flatten()) == 28*28:
            SZ = (28, 28)
            CMAP = 'gray'
        elif len(r[0].flatten()) == 224*224*3:
            SZ = (224, 224, 3)
            CMAP = None
        else:
            raise Exception("cannot handle input image")
        #lab.imshow(denorm(imorig[0]).numpy().transpose((1,2,0)))
        #lab.show()

        # traverse this batch of images.
        for i in range(xr.size(0)):

            if not normimg:
                imo = imorig[i].detach().cpu().squeeze().view(*SZ).numpy()
                ima = xr[i].detach().cpu().squeeze().view(*SZ).numpy()
                imr = r[i].detach().cpu().squeeze().view(*SZ).numpy()
            else:
                imo = denorm(imorig[i]).detach().cpu().squeeze().numpy().transpose((1,2,0))
                ima = denorm(xr[i]).detach().cpu().squeeze().numpy().transpose((1,2,0))
                imr = xdnorm(r[i]).detach().cpu().squeeze().numpy().transpose((1,2,0))
            tidx = targetidx[i]
            assert(len(tidx) == 1)
            if not normimg:
                imq = loader.dataset[tidx[0].item()][0].squeeze().view(*SZ).numpy()
            else:
                imq = denorm(loader.dataset[tidx[0].item()][0]).squeeze().numpy().transpose((1,2,0))
            imqv = cddts[tidx[0]]
            if metric == 'C':
                qcdist = (1-th.mm(cddts, imqv.view(-1, 1))).flatten()
                qcsort = qcdist.argsort(descending=False)
                qcrank = qcdist.argsort(descending=False).argsort(descending=False)
                qcadist = (1 - th.dot(imqv, outadv[i]))
                qcarank = (qcadist >= qcdist).sum()
                qcodist = (1 - th.dot(imqv, outorig[i]))
                qcorank = (qcodist >= qcdist).sum()
                #print(qcdist.shape)
            else:
                qcdist = (cddts - imqv.view(1, -1)).norm(2,dim=1)
                qcsort = qcdist.argsort(); qcrank = qcdist.argsort().argsort()
                qcadist = (outadv[i] - imqv).norm(2)
                qcarank = (qcadist >= qcdist).sum()
                qcodist = (outorig[i] - imqv).norm(2)
                qcorank = (qcodist >= qcdist).sum()
            print('target (Q) idx = ', tidx)
            print(qcadist, qcarank, qcodist, qcorank)

            lab.figure(figsize=(21/3,9/3))
            # row 1
            lab.subplot(3, 10, 2); lab.imshow(imo, cmap=CMAP)
            lab.title('$c$'); lab.axis('off')

            lab.subplot(3, 10, 4); lab.imshow(imr, cmap=CMAP)
            lab.title('$r$'); lab.axis('off')

            lab.subplot(3, 10, 6); lab.imshow(ima, cmap=CMAP)
            lab.title('$\\tilde{c}=c+r$'); lab.axis('off')

            # row 2
            lab.subplot(3, 10, 11); lab.imshow(imq, cmap=CMAP)
            lab.title('$q$'); lab.axis('off')

            ID = 0
            imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 12); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('0')
            imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 13); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('1')
            imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 14); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('2')
            imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 15); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('3')
            imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 16); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('4')
            #lab.subplot(3, 10, 17); lab.imshow(imx, cmap=CMAP); lab.axis('off')
            lab.subplot(3, 10, 18);
            idx = max(0,qcorank-1)
            imx = loader.dataset[qcsort[idx].item()][0].view(*SZ).numpy()
            if normimg: imx = denorm(loader.dataset[qcsort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{idx}')
            lab.subplot(3, 10, 19);
            idx = qcorank
            lab.imshow(imo, cmap=CMAP);
            lab.axis('off'); lab.title(f'{idx}')
            lab.subplot(3, 10, 20)
            idx = min(cddts.size(0)-1, qcorank+2)
            imx = loader.dataset[qcsort[idx].item()][0].view(*SZ).numpy()
            if normimg: imx = denorm(loader.dataset[qcsort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{idx-1}')

            # row 3
            lab.subplot(3, 10, 21); lab.imshow(imq, cmap=CMAP)
            lab.title('$q$'); lab.axis('off')

            ID = 0
            if qcarank == 0:
                imx = ima
            else:
                imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
                if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 22); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('0')
            if qcarank == 1:
                imx = ima
            else:
                imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
                if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 23); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('1')
            if qcarank == 2:
                imx = ima
            else:
                imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
                if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 24); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('2')
            if qcarank == 3:
                imx = ima
            else:
                imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
                if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 25); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('3')
            if qcarank == 4:
                imx = ima
            else:
                imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
                if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 26); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('4')
            #lab.subplot(3, 10, 17); lab.imshow(imx, cmap=CMAP); lab.axis('off')
            idx = max(0,qcarank-1)
            if idx != 0:
                lab.subplot(3, 10, 28);
                if not normimg:
                    imx = loader.dataset[qcsort[idx].item()][0].view(*SZ).numpy()
                else:
                    imx = denorm(loader.dataset[qcsort[idx].item()][0]).numpy().transpose((1,2,0))
                lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{idx}')
            lab.subplot(3, 10, 29);
            idx = qcarank
            lab.imshow(ima, cmap=CMAP);
            lab.axis('off'); lab.title(f'{idx}')
            lab.subplot(3, 10, 30)
            idx = min(cddts.size(0)-1, qcarank)
            if not normimg:
                imx = loader.dataset[qcsort[idx].item()][0].view(*SZ).numpy()
            else:
                imx = denorm(loader.dataset[qcsort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{idx+1}')

            lab.subplots_adjust(hspace=0.1, wspace=0.1)
            if not normimg:
                lab.savefig('x.svg')
            else:
                lab.savefig('x.svg', dpi=300)
            lab.show()

    @staticmethod
    def vis_caminus(xr, r, imorig, # perturbed images, perturb, original images
            outorig, outadv, # original/adversarial distance matrix
            targetidx, # Q:idx for CA, C:idx for QA
            loader, # for loading images outside of the batch
            cddts,
            *,
            normimg=False, # for SOP
            metric=None,
            ):
        '''
        visualize CA- effects
        '''
        if len(r[0].flatten()) == 28*28:
            SZ = (28, 28)
            CMAP = 'gray'
        elif len(r[0].flatten()) == 224*224*3:
            SZ = (224, 224, 3)
            CMAP = None
        else:
            raise Exception("cannot handle input image")

        # traverse this batch of images.
        for i in range(xr.size(0)):

            if not normimg:
                imo = imorig[i].detach().cpu().squeeze().view(28, 28).numpy()
                ima = xr[i].detach().cpu().squeeze().view(28, 28).numpy()
                imr = r[i].detach().cpu().squeeze().view(28, 28).numpy()
            else:
                imo = denorm(imorig[i]).detach().cpu().squeeze().numpy().transpose((1,2,0))
                ima = denorm(xr[i]).detach().cpu().squeeze().numpy().transpose((1,2,0))
                imr = xdnorm(r[i]).detach().cpu().squeeze().numpy().transpose((1,2,0))
            tidx = targetidx[i]
            assert(len(tidx) == 1)
            if not normimg:
                imq = loader.dataset[tidx[0].item()][0].squeeze().view(28, 28).numpy()
            else:
                imq = denorm(loader.dataset[tidx[0].item()][0]).squeeze().numpy().transpose((1,2,0))
            imqv = cddts[tidx[0]]
            if metric == 'C':
                qcdist = (1-th.mm(cddts, imqv.view(-1, 1))).flatten()
                qcsort = qcdist.argsort(descending=False)
                acrank = qcdist.argsort(descending=False).argsort(descending=False)
                qcadist = (1 - th.dot(imqv, outadv[i]))
                qcarank = (qcadist >= qcdist).sum() - 1
                qcodist = (1 - th.dot(imqv, outorig[i]))
                qcorank = (qcodist >= qcdist).sum() - 1
                #print(qcdist.shape)
            else:
                qcdist = (cddts - imqv.view(1, -1)).norm(2,dim=1)
                qcsort = qcdist.argsort(); qcrank = qcdist.argsort().argsort()
                qcadist = (outadv[i] - imqv).norm(2)
                qcarank = (qcadist >= qcdist).sum() - 1
                qcodist = (outorig[i] - imqv).norm(2)
                qcorank = (qcodist >= qcdist).sum() - 1
            print('target (Q) idx = ', tidx)

            lab.figure(figsize=(21/3,9/3))
            # row 1
            lab.subplot(3, 10, 2); lab.imshow(imo, cmap=CMAP)
            lab.title('$c$'); lab.axis('off')

            lab.subplot(3, 10, 4); lab.imshow(imr, cmap=CMAP)
            lab.title('$r$'); lab.axis('off')

            lab.subplot(3, 10, 6); lab.imshow(ima, cmap=CMAP)
            lab.title('$\\tilde{c}=c+r$'); lab.axis('off')

            # row 2
            lab.subplot(3, 10, 11); lab.imshow(imq, cmap=CMAP)
            lab.title('$q$'); lab.axis('off')

            ID = 0
            imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 12); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('0')
            imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 13); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('1')
            imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 14); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('2')
            imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 15); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('3')
            imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 16); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('4')
            #lab.subplot(3, 10, 17); lab.imshow(imx, cmap=CMAP); lab.axis('off')
            lab.subplot(3, 10, 18);
            idx = max(0,qcorank-1)
            imx = loader.dataset[qcsort[idx].item()][0].view(*SZ).numpy()
            if normimg: imx = denorm(loader.dataset[qcsort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{idx}')
            lab.subplot(3, 10, 19);
            idx = qcorank
            lab.imshow(imo, cmap=CMAP);
            lab.axis('off'); lab.title(f'{idx}')
            lab.subplot(3, 10, 20)
            idx = min(cddts.size(0)-1, qcorank+2)
            imx = loader.dataset[qcsort[idx].item()][0].view(*SZ).numpy()
            if normimg: imx = denorm(loader.dataset[qcsort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{idx-1}')

            # row 3
            lab.subplot(3, 10, 21); lab.imshow(imq, cmap=CMAP)
            lab.title('$q$'); lab.axis('off')

            ID = 0
            if qcorank == 0:
                ID+=1
            imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 22); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('0')
            if qcorank == 1:
                ID+=1
            imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 23); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('1')
            if qcorank == 2:
                ID+=1
            imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 24); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('2')
            if qcorank == 3:
                ID+=1
            imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 25); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('3')
            if qcorank == 4:
                ID+=1
            imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 26); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('4')
            #lab.subplot(3, 10, 17); lab.imshow(imx, cmap=CMAP); lab.axis('off')
            lab.subplot(3, 10, 28);
            idx = max(0,qcarank-1)
            imx = loader.dataset[qcsort[idx].item()][0].view(*SZ).numpy()
            if normimg: imx = loader.dataset[qcsort[idx].item()][0].numpy().transpose((1,2,0))
            lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{idx}')
            lab.subplot(3, 10, 29);
            idx = qcarank
            lab.imshow(ima, cmap=CMAP);
            lab.axis('off'); lab.title(f'{idx}')
            idx = min(cddts.size(0)-1, qcarank)
            if idx != cddts.size(0)-1:
                lab.subplot(3, 10, 30)
                imx = loader.dataset[qcsort[idx].item()][0].view(*SZ).numpy()
                if normimg: imx = denorm(loader.dataset[qcsort[idx].item()][0]).numpy().transpose((1,2,0))
                lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{idx+1}')

            lab.subplots_adjust(hspace=0.1, wspace=0.1)
            if not normimg:
                lab.savefig('x.svg')
            else:
                lab.savefig('x.svg', dpi=300)
            lab.show()

    @staticmethod
    def vis_qaplus(xr, r, imorig,
            outorig, outadv,
            targetidx,
            loader,
            cddts,
            *,
            normimg=False,
            metric=None,
            ):
        '''
        visualiza QA+ effects
        '''
        if len(r[0].flatten()) == 28*28:
            SZ = (28, 28)
            CMAP = 'gray'
        elif len(r[0].flatten()) == 224*224*3:
            SZ = (224, 224, 3)
            CMAP = None
        else:
            raise Exception("cannot handle input image")

        for i in range(xr.size(0)):

            if not normimg:
                imo = imorig[i].detach().cpu().squeeze().view(*SZ).numpy()
                ima = xr[i].detach().cpu().squeeze().view(*SZ).numpy()
                imr = r[i].detach().cpu().squeeze().view(*SZ).numpy()
            else:
                imo = denorm(imorig[i]).detach().cpu().squeeze().numpy().transpose((1,2,0))
                ima = denorm(xr[i]).detach().cpu().squeeze().numpy().transpose((1,2,0))
                imr = xdnorm(r[i]).detach().cpu().squeeze().numpy().transpose((1,2,0))
            imov = outorig[i]
            imav = outadv[i]
            tidx = targetidx[i]
            assert(len(tidx) == 1)
            if not normimg:
                imc = loader.dataset[tidx[0].item()][0].squeeze().view(*SZ).numpy()
            else:
                imc = denorm(loader.dataset[tidx[0].item()][0]).squeeze().numpy().transpose((1,2,0))
            imcv = cddts[tidx[0]]
            if metric == 'C':
                qcdist = (1-th.mm(cddts, imov.view(-1, 1))).flatten()
                qcsort = qcdist.argsort(descending=False)
                qcrank = qcdist.argsort(descending=False).argsort(descending=False)
                acdist = (1-th.mm(cddts, imav.view(-1, 1))).flatten()
                acsort = acdist.argsort()
                acrank = acdist.argsort().argsort()

                qcdist_ = (1 - th.dot(outorig[i], imcv))
                qcrank_ = (qcdist_ > qcdist).sum()
                acdist_ = (1 - th.dot(outadv[i], imcv))
                acrank_ = (acdist_ > acdist).sum()
            #    #print(qcdist.shape)
            else:
                qcdist = (cddts - imov.view(1, -1)).norm(2, dim=1).flatten()
                qcsort = qcdist.argsort()
                qcrank = qcdist.argsort().argsort()
                acdist = (cddts - imav.view(1, -1)).norm(2, dim=1).flatten()
                acsort = acdist.argsort()
                acrank = acdist.argsort().argsort()

                qcdist_ = (imov - imcv).norm(2)
                qcrank_ = (qcdist_ > qcdist).sum()
                acdist_ = (imav - imcv).norm(2)
                acrank_ = (acdist_ > acdist).sum()
            print('target (C) idx = ', tidx)
            print('orig dist/rank', qcdist_, qcrank_)
            print('adv dist/rank', acdist_, acrank_)

            lab.figure(figsize=(21/3,9/3))
            # row 1
            lab.subplot(3, 10, 2); lab.imshow(imo, cmap=CMAP)
            lab.title('$q$'); lab.axis('off')

            lab.subplot(3, 10, 4); lab.imshow(imr, cmap=CMAP)
            lab.title('$r$'); lab.axis('off')

            lab.subplot(3, 10, 6); lab.imshow(ima, cmap=CMAP)
            lab.title('$\\tilde{q}=q+r$'); lab.axis('off')

            # row 2
            lab.subplot(3, 10, 11); lab.imshow(imo, cmap=CMAP)
            lab.title('$q$'); lab.axis('off')

            ID = 0
            imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 12); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('0')
            imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 13); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('1')
            imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 14); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('2')
            imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 15); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('3')
            imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 16); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('4')
            #lab.subplot(3, 10, 17); lab.imshow(imx, cmap=CMAP); lab.axis('off')
            lab.subplot(3, 10, 18);
            idx = max(0,qcrank_-1)
            if not normimg:
                imx = loader.dataset[qcsort[idx]][0].view(*SZ).numpy()
            else:
                imx = denorm(loader.dataset[qcsort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{idx}')
            lab.subplot(3, 10, 19);
            idx = qcrank_
            lab.imshow(imc, cmap=CMAP);
            lab.axis('off'); lab.title(f'{idx}')
            lab.subplot(3, 10, 20)
            idx = min(cddts.size(0)-1, qcrank_+2)
            if not normimg:
                imx = loader.dataset[qcsort[idx]][0].view(*SZ).numpy()
            else:
                imx = denorm(loader.dataset[qcsort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{idx-1}')

            # row 3
            lab.subplot(3, 10, 21); lab.imshow(ima, cmap=CMAP)
            lab.title('$\\tilde{q}$'); lab.axis('off')

            ID = 0
            imx = loader.dataset[acsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[acsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 22); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('0')
            imx = loader.dataset[acsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[acsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 23); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('1')
            imx = loader.dataset[acsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[acsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 24); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('2')
            imx = loader.dataset[acsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[acsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 25); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('3')
            imx = loader.dataset[acsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[acsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 26); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('4')
            ##lab.subplot(3, 10, 17); lab.imshow(imx, cmap=CMAP); lab.axis('off')
            lab.subplot(3, 10, 28);
            idx = max(0,acrank_-1)
            if not normimg:
                imx = loader.dataset[acsort[idx]][0].view(*SZ).numpy()
            else:
                imx = denorm(loader.dataset[acsort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{idx}')
            lab.subplot(3, 10, 29);
            idx = acrank_
            lab.imshow(imc, cmap=CMAP);
            lab.axis('off'); lab.title(f'{idx}')
            lab.subplot(3, 10, 30)
            idx = min(cddts.size(0)-1, acrank_+2)
            if not normimg:
                imx = loader.dataset[acsort[idx]][0].view(*SZ).numpy()
            else:
                imx = denorm(loader.dataset[acsort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{idx-1}')

            lab.subplots_adjust(hspace=0.1, wspace=0.1)
            if not normimg:
                lab.savefig('x.svg')
            else:
                lab.savefig('x.svg', dpi=300)
            lab.show()

    @staticmethod
    def vis_qaminus(xr, r, imorig,
            outorig, outadv,
            targetidx,
            loader,
            cddts,
            *,
            normimg=False,
            metric=None,
            ):
        '''
        visualiza QA+ effects
        '''
        if len(r[0].flatten()) == 28*28:
            SZ = (28, 28)
            CMAP = 'gray'
        elif len(r[0].flatten()) == 224*224*3:
            SZ = (224, 224, 3)
            CMAP = None
        else:
            raise Exception("cannot handle input image")

        for i in range(xr.size(0)):

            if not normimg:
                imo = imorig[i].detach().cpu().squeeze().view(*SZ).numpy()
                ima = xr[i].detach().cpu().squeeze().view(*SZ).numpy()
                imr = r[i].detach().cpu().squeeze().view(*SZ).numpy()
            else:
                imo = denorm(imorig[i]).detach().cpu().squeeze().numpy().transpose((1,2,0))
                ima = denorm(xr[i]).detach().cpu().squeeze().numpy().transpose((1,2,0))
                imr = xdnorm(r[i]).detach().cpu().squeeze().numpy().transpose((1,2,0))
            imov = outorig[i]
            imav = outadv[i]
            tidx = targetidx[i]
            assert(len(tidx) == 1)
            if not normimg:
                imc = loader.dataset[tidx[0]][0].squeeze().view(*SZ).numpy()
            else:
                imc = denorm(loader.dataset[tidx[0].item()][0].squeeze()).numpy().transpose((1,2,0))
            imcv = cddts[tidx[0]]
            if metric == 'C':
                qcdist = (1-th.mm(cddts, imov.view(-1, 1))).flatten()
                qcsort = qcdist.argsort(descending=False)
                acrank = qcdist.argsort(descending=False).argsort(descending=False)
                acdist = (1-th.mm(cddts, imav.view(-1, 1))).flatten()
                acsort = acdist.argsort()
                acrank = acdist.argsort().argsort()

                qcdist_ = (1 - th.dot(outorig[i], imcv))
                qcrank_ = (qcdist_ >= qcdist).sum() - 1
                acdist_ = (1 - th.dot(outadv[i], imcv))
                acrank_ = (acdist_ >= acdist).sum()
            #    #print(qcdist.shape)
            else:
                qcdist = (cddts - imov.view(1, -1)).norm(2, dim=1).flatten()
                qcsort = qcdist.argsort()
                qcrank = qcdist.argsort().argsort()
                acdist = (cddts - imav.view(1, -1)).norm(2, dim=1).flatten()
                acsort = acdist.argsort()
                acrank = acdist.argsort().argsort()

                qcdist_ = (imov - imcv).norm(2)
                qcrank_ = (qcdist_ >= qcdist).sum() - 1
                acdist_ = (imav - imcv).norm(2)
                acrank_ = (acdist_ >= acdist).sum() - 1
            print('target (C) idx = ', tidx)
            print('orig dist/rank', qcdist_, qcrank_)
            print('adv dist/rank', acdist_, acrank_)

            lab.figure(figsize=(21/3,9/3))
            # row 1
            lab.subplot(3, 10, 2); lab.imshow(imo, cmap=CMAP)
            lab.title('$q$'); lab.axis('off')

            lab.subplot(3, 10, 4); lab.imshow(imr, cmap=CMAP)
            lab.title('$r$'); lab.axis('off')

            lab.subplot(3, 10, 6); lab.imshow(ima, cmap=CMAP)
            lab.title('$\\tilde{q}=q+r$'); lab.axis('off')

            # row 2
            lab.subplot(3, 10, 11); lab.imshow(imo, cmap=CMAP)
            lab.title('$q$'); lab.axis('off')

            ID = 0
            if qcrank_ == 0:
                imx = imc; ID+=1
            else:
                imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
                if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 12); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('0')
            if qcrank_ == 1:
                imx = imc; ID+=1
            else:
                imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
                if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 13); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('1')
            if qcrank_ == 2:
                imx = imc; ID+=1
            else:
                imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
                if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 14); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('2')
            if qcrank_ == 3:
                imx = imc; ID+=1
            else:
                imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
                if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 15); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('3')
            if qcrank_ == 4:
                imx = imc; ID+=1
            else:
                imx = loader.dataset[qcsort[ID].item()][0].view(*SZ).numpy(); ID+=1
                if normimg: imx = denorm(loader.dataset[qcsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 16); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('4')
            #lab.subplot(3, 10, 17); lab.imshow(imx, cmap=CMAP); lab.axis('off')
            lab.subplot(3, 10, 18);
            idx = max(0,qcrank_-1)
            if not normimg:
                imx = loader.dataset[qcsort[idx]][0].view(*SZ).numpy()
            else:
                imx = denorm(loader.dataset[qcsort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{idx}')
            lab.subplot(3, 10, 19);
            idx = qcrank_
            lab.imshow(imc, cmap=CMAP);
            lab.axis('off'); lab.title(f'{idx}')
            lab.subplot(3, 10, 20)
            idx = min(cddts.size(0)-1, qcrank_+1)
            if not normimg:
                imx = loader.dataset[qcsort[idx]][0].view(*SZ).numpy()
            else:
                imx = denorm(loader.dataset[qcsort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{idx}')

            # row 3
            lab.subplot(3, 10, 21); lab.imshow(ima, cmap=CMAP)
            lab.title('$\\tilde{q}$'); lab.axis('off')

            ID = 0
            imx = loader.dataset[acsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[acsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 22); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('0')
            imx = loader.dataset[acsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[acsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 23); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('1')
            imx = loader.dataset[acsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[acsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 24); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('2')
            imx = loader.dataset[acsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[acsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 25); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('3')
            imx = loader.dataset[acsort[ID].item()][0].view(*SZ).numpy(); ID+=1
            if normimg: imx = denorm(loader.dataset[acsort[ID-1].item()][0]).numpy().transpose((1,2,0))
            lab.subplot(3, 10, 26); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title('4')
            ##lab.subplot(3, 10, 17); lab.imshow(imx, cmap=CMAP); lab.axis('off')
            lab.subplot(3, 10, 28);
            idx = max(0,acrank_-1)
            if not normimg:
                imx = loader.dataset[acsort[idx]][0].view(*SZ).numpy()
            else:
                imx = denorm(loader.dataset[acsort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{idx}')
            lab.subplot(3, 10, 29);
            lab.imshow(imc, cmap=CMAP);
            lab.axis('off'); lab.title(f'{acrank_}')
            lab.subplot(3, 10, 30)
            idx = min(cddts.size(0)-1, acrank_+1)
            if not normimg:
                imx = loader.dataset[acsort[idx]][0].view(*SZ).numpy()
            else:
                imx = denorm(loader.dataset[acsort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{idx}')

            lab.subplots_adjust(hspace=0.1, wspace=0.1)
            if not normimg:
                lab.savefig('x.svg')
            else:
                lab.savefig('x.svg', dpi=300)
            lab.show()

    @staticmethod
    def vis_es():
        raise NotImplementedError  # obsolete
        if normimg:
            r = r.mul(IMstd[:,None,None])
        print(_fgcGrey('r>', 'Min', '%.3f'%r.min().item(),
                'Max', '%.3f'%r.max().item(),
                'Mean', '%.3f'%r.mean().item(),
                'L0', '%.3f'%r.norm(0).item(),
                'L1', '%.3f'%r.norm(1).item(),
                'L2', '%.3f'%r.norm(2).item()))
        # print the image if batchsize is 1
        if xr.shape[0] == 1:
            im = images_orig.cpu().squeeze().view(28, 28).numpy()
            advim = xr.detach().cpu().squeeze().view(28, 28).numpy()
            advpr = r.detach().cpu().squeeze().view(28, 28).numpy()
            lab.subplot(2,8,1)
            lab.imshow(im, cmap='gray')
            lab.xlabel(f'Orig {labels.item()}')
            lab.colorbar()
            lab.subplot(2,8,8+2)
            lab.imshow(advpr, cmap='gray')
            lab.colorbar()
            lab.xlabel('Perturbation')
            lab.subplot(2,8,8+3)
            lab.imshow(advim, cmap='gray')
            lab.xlabel('Adv')
            lab.colorbar()
            for i in (4,5,6,7,8):
                im_i = loader.dataset[knnsearch_orig[i-3]][0].squeeze().view(28, 28).numpy()
                im_l = loader.dataset[knnsearch_orig[i-3]][1].squeeze().item()
                lab.subplot(2,8,i)
                lab.imshow(im_i, cmap='gray')
                lab.xlabel(f'{im_l}')

                im_i = loader.dataset[knnsearch_adv[i-4]][0].squeeze().view(28, 28).numpy()
                im_l = loader.dataset[knnsearch_adv[i-4]][1].squeeze().item()
                lab.subplot(2, 8, i+8)
                lab.imshow(im_i, cmap='gray')
                lab.xlabel(f'{im_l}')
            lab.show()

    @staticmethod
    def vis_caplus2(xr, r, imorig, # perturbed images, perturb, original images
            outorig, outadv, # original/adversarial distance matrix
            targetidx, # Q:idx for CA, C:idx for QA
            loader, # for loading images outside of the batch
            cddts,
            *,
            normimg=False, # for SOP
            metric=None,
            ):
        '''
        visualize CA+ effects, note, this is the w=2 version, i.e. we choose Q={q1,q2}
        '''
        assert(len(r[0].flatten()) == 224*224*3)
        assert(normimg == True)
        assert(len(targetidx[0]) == 2)
        SZ = (224, 224, 3)
        CMAP = None
        #lab.imshow(denorm(imorig[0]).numpy().transpose((1,2,0)))
        #lab.show()

        # traverse this batch of images.
        for i in range(xr.size(0)):
            # adversarial candidate
            imo = denorm(imorig[i]).detach().cpu().squeeze().numpy().transpose((1,2,0))
            ima = denorm(xr[i]).detach().cpu().squeeze().numpy().transpose((1,2,0))
            imr = xdnorm(r[i]).detach().cpu().squeeze().numpy().transpose((1,2,0))
            # idx of Q={...}
            tidx = targetidx[i]
            imq1 = denorm(loader.dataset[tidx[0].item()][0]).squeeze().numpy().transpose((1,2,0))
            imq2 = denorm(loader.dataset[tidx[1].item()][0]).squeeze().numpy().transpose((1,2,0))
            imq1v = cddts[tidx[0]]
            imq2v = cddts[tidx[1]]
            if metric == 'E':
                # q1 and candidates
                q1cdist = (cddts - imq1v.view(1, -1)).norm(2,dim=1)
                q1csort = q1cdist.argsort()
                q1odist_ = (outorig[i] - imq1v).norm(2)
                q1orank_ = (q1odist_ >= q1cdist).sum()
                q1adist_ = (outadv[i] - imq1v).norm(2)
                q1arank_ = (q1adist_ >= q1cdist).sum()
                # q2 and candidates
                q2cdist = (cddts - imq2v.view(1, -1)).norm(2, dim=1)
                q2csort = q2cdist.argsort()
                q2odist_ = (outorig[i] - imq2v).norm(2)
                q2orank_ = (q2odist_ >= q2cdist).sum()
                q2adist_ = (outadv[i] - imq2v).norm(2)
                q2arank_ = (q2adist_ >= q2cdist).sum()
            print('target (Q) idx = ', tidx)

            lab.figure(figsize=(21/3,16/3))
            # row 1: c, r, c+r
            lab.subplot(5, 10, 2); lab.imshow(imo, cmap=CMAP); lab.title('$c$'); lab.axis('off')
            lab.subplot(5, 10, 4); lab.imshow(imr, cmap=CMAP); lab.title('$r$'); lab.axis('off')
            lab.subplot(5, 10, 6); lab.imshow(ima, cmap=CMAP); lab.title('$\\tilde{c}=c+r$'); lab.axis('off')

            # row 2: q1, original
            lab.subplot(5, 10, 11); lab.imshow(imq1, cmap=CMAP); lab.title('$q_1$'); lab.axis('off')
            ID = 0
            for j in range(5):
                imx = denorm(loader.dataset[q1csort[ID].item()][0]).numpy().transpose((1,2,0))
                ID+=1
                lab.subplot(5, 10, 12+j); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{j}')
            #lab.subplot(3, 10, 17); lab.imshow(imx, cmap=CMAP); lab.axis('off')
            lab.subplot(5, 10, 18);
            idx = max(0,q1orank_-1)
            imx = denorm(loader.dataset[q1csort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{idx}')
            #
            lab.subplot(5, 10, 19);
            idx = q1orank_
            lab.imshow(imo, cmap=CMAP); lab.axis('off'); lab.title(f'{idx}')
            #
            lab.subplot(5, 10, 20)
            idx = min(cddts.size(0)-1, q1orank_+1)
            imx = denorm(loader.dataset[q1csort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{idx}')

            # row 4: q2, original
            lab.subplot(5, 10, 31); lab.imshow(imq2, cmap=CMAP); lab.title('$q_2$'); lab.axis('off')
            ID = 0
            for j in range(5):
                imx = denorm(loader.dataset[q2csort[ID].item()][0]).numpy().transpose((1,2,0))
                ID += 1
                lab.subplot(5, 10, 32+j); lab.imshow(imx); lab.axis('off'); lab.title(f'{j}')
            # split line
            lab.subplot(5, 10, 38)
            idx = max(0, q2orank_ - 1)
            imx = denorm(loader.dataset[q2csort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx); lab.axis('off'); lab.title(f'{idx}')
            #
            lab.subplot(5, 10, 39)
            idx = q2orank_; lab.imshow(imo); lab.axis('off'); lab.title(f'{idx}')
            #
            lab.subplot(5, 10, 40)
            idx = min(cddts.size(0) - 1, q2orank_ + 1)
            imx = denorm(loader.dataset[q2csort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx); lab.axis('off'); lab.title(f'{idx}')

            # row 3: q1, adversarial candidate
            lab.subplot(5, 10, 21); lab.imshow(imq1, cmap=CMAP); lab.title('$q_1$'); lab.axis('off')
            ID = 0
            for j in range(5):
                if q1arank_ == j:
                    imx = ima
                else:
                    imx = denorm(loader.dataset[q1csort[ID].item()][0]).numpy().transpose((1,2,0))
                    ID += 1
                lab.subplot(5, 10, 22+j); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{j}')
            #lab.subplot(3, 10, 17); lab.imshow(imx, cmap=CMAP); lab.axis('off')
            idx = q1arank_-1
            if idx >= 0:
                lab.subplot(5, 10, 28);
                imx = denorm(loader.dataset[q1csort[idx].item()][0]).numpy().transpose((1,2,0))
                lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{idx}')
            #
            idx = q1arank_; lab.subplot(5, 10, 29);
            lab.imshow(ima, cmap=CMAP); lab.axis('off'); lab.title(f'{idx}')
            #
            lab.subplot(5, 10, 30)
            idx = q1arank_
            imx = denorm(loader.dataset[q1csort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{idx+1}')

            # row 5: q2, adversarial candidate
            lab.subplot(5, 10, 41); lab.imshow(imq2); lab.title('$q_2$'); lab.axis('off')
            ID = 0
            for j in range(5):
                if q2arank_ == j:
                    imx = ima
                else:
                    imx = denorm(loader.dataset[q2csort[ID].item()][0]).numpy().transpose((1,2,0))
                    ID += 1
                lab.subplot(5, 10, 42+j); lab.imshow(imx); lab.axis('off'); lab.title(f'{j}')
            # gap
            idx = q2arank_-1
            if idx >= 0:
                lab.subplot(5, 10, 48);
                imx = denorm(loader.dataset[q2csort[idx].item()][0]).numpy().transpose((1,2,0))
                lab.imshow(imx); lab.axis('off'); lab.title(f'{idx}')
            #
            idx = q2arank_; lab.subplot(5, 10, 49); 
            lab.imshow(ima); lab.axis('off'); lab.title(f'{idx}')
            #
            idx = q2arank_; lab.subplot(5, 10, 50);
            imx = denorm(loader.dataset[q2csort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx); lab.axis('off'); lab.title(f'{idx+1}')

            # typesetting, and dump
            lab.subplots_adjust(hspace=0.1, wspace=0.1)
            lab.savefig('x.svg', dpi=300)
            lab.show()

    @staticmethod
    def vis_qaplus2(xr, r, imorig,
            outorig, outadv,
            targetidx,
            loader,
            cddts,
            *,
            normimg=False,
            metric=None,
            ):
        '''
        visualiza QA+ effects, QA+ M=2
        '''
        SZ = (224, 224, 3)
        CMAP = None
        for i in range(xr.size(0)):
            # adversarial query
            imo = denorm(imorig[i]).detach().cpu().squeeze().numpy().transpose((1,2,0))
            ima = denorm(xr[i]).detach().cpu().squeeze().numpy().transpose((1,2,0))
            imr = xdnorm(r[i]).detach().cpu().squeeze().numpy().transpose((1,2,0))
            imov = outorig[i]
            imav = outadv[i]
            # candidate set C={...}
            tidx = targetidx[i]
            imc1 = denorm(loader.dataset[tidx[0].item()][0]).squeeze().numpy().transpose((1,2,0))
            imc1v = cddts[tidx[0]]
            imc2 = denorm(loader.dataset[tidx[1].item()][0]).squeeze().numpy().transpose((1,2,0))
            imc2v = cddts[tidx[1]]
            if metric == 'E':
                # q/q! and X
                qcdist = (cddts - imov.view(1, -1)).norm(2, dim=1).flatten()
                qcsort = qcdist.argsort()
                acdist = (cddts - imav.view(1, -1)).norm(2, dim=1).flatten()
                acsort = acdist.argsort()
                # q/q~ and c1
                qc1dist_ = (imov - imc1v).norm(2)
                qc1rank_ = (qc1dist_ > qcdist).sum()
                ac1dist_ = (imav - imc1v).norm(2)
                ac1rank_ = (ac1dist_ > acdist).sum()
                # q/q~ and c2
                qc2dist_ = (imov - imc2v).norm(2)
                qc2rank_ = (qc2dist_ > qcdist).sum()
                ac2dist_ = (imav - imc2v).norm(2)
                ac2rank_ = (ac2dist_ > acdist).sum()
            print('target (C) idx = ', tidx)

            lab.figure(figsize=(21/3,16/3))
            # row 1: q, r, q+r
            lab.subplot(5, 10, 2); lab.imshow(imo); lab.title('$q$'); lab.axis('off')
            lab.subplot(5, 10, 4); lab.imshow(imr); lab.title('$r$'); lab.axis('off')
            lab.subplot(5, 10, 6); lab.imshow(ima); lab.title('$\\tilde{q}=q+r$'); lab.axis('off')

            # row 2; q, c1, original
            lab.subplot(5, 10, 11); lab.imshow(imo); lab.title('$q$'); lab.axis('off')
            ID = 0
            for j in range(5):
                imx = denorm(loader.dataset[qcsort[ID].item()][0]).numpy().transpose((1,2,0))
                ID+=1
                lab.subplot(5, 10, 12+j); lab.imshow(imx); lab.axis('off'); lab.title(f'{j}')
            #lab.subplot(3, 10, 17); lab.imshow(imx, cmap=CMAP); lab.axis('off')
            idx = max(0,qc1rank_-1); lab.subplot(5, 10, 18);
            imx = denorm(loader.dataset[qcsort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{idx}')
            #
            idx = qc1rank_; lab.subplot(5, 10, 19);
            lab.imshow(imc1, cmap=CMAP); lab.axis('off'); lab.title(f'{idx}')
            #
            idx = min(cddts.size(0)-1, qc1rank_+2); lab.subplot(5, 10, 20)
            imx = denorm(loader.dataset[qcsort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx); lab.axis('off'); lab.title(f'{idx-1}')

            # row 4: q, c2, original
            lab.subplot(5, 10, 31); lab.imshow(imo); lab.title('$q$'); lab.axis('off')
            ID = 0
            for j in range(5):
                imx = denorm(loader.dataset[qcsort[ID].item()][0]).numpy().transpose((1,2,0))
                ID+=1
                lab.subplot(5, 10, 32+j); lab.imshow(imx); lab.axis('off'); lab.title(f'{j}')
            #lab.subplot(3, 10, 17); lab.imshow(imx, cmap=CMAP); lab.axis('off')
            idx = max(0,qc2rank_-1); lab.subplot(5, 10, 38);
            imx = denorm(loader.dataset[qcsort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx); lab.axis('off'); lab.title(f'{idx}')
            #
            idx = qc2rank_; lab.subplot(5, 10, 39);
            lab.imshow(imc2); lab.axis('off'); lab.title(f'{idx}')
            #
            idx = min(cddts.size(0)-1, qc2rank_+2); lab.subplot(5, 10, 40)
            imx = denorm(loader.dataset[qcsort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx); lab.axis('off'); lab.title(f'{idx-1}')

            # row 3: q, c1, adversarial
            lab.subplot(5, 10, 21); lab.imshow(ima, cmap=CMAP); lab.title('$\\tilde{q}$'); lab.axis('off')
            ID = 0
            for j in range(5):
                imx = denorm(loader.dataset[acsort[ID].item()][0]).numpy().transpose((1,2,0))
                ID += 1
                lab.subplot(5, 10, 22+j); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{j}')
            ##lab.subplot(3, 10, 17); lab.imshow(imx, cmap=CMAP); lab.axis('off')
            idx = max(0,ac1rank_-1); lab.subplot(5, 10, 28);
            imx = denorm(loader.dataset[acsort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx); lab.axis('off'); lab.title(f'{idx}')
            #
            idx = ac1rank_; lab.subplot(5, 10, 29);
            lab.imshow(imc1); lab.axis('off'); lab.title(f'{idx}')
            #
            idx = min(cddts.size(0)-1, ac1rank_+2); lab.subplot(5, 10, 30)
            imx = denorm(loader.dataset[acsort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{idx-1}')

            # row 5: q, c2, adversarial
            lab.subplot(5, 10, 41); lab.imshow(ima, cmap=CMAP); lab.title('$\\tilde{q}$'); lab.axis('off')
            ID = 0
            for j in range(5):
                imx = denorm(loader.dataset[acsort[ID].item()][0]).numpy().transpose((1,2,0))
                ID += 1
                lab.subplot(5, 10, 42+j); lab.imshow(imx, cmap=CMAP); lab.axis('off'); lab.title(f'{j}')
            ##lab.subplot(3, 10, 17); lab.imshow(imx, cmap=CMAP); lab.axis('off')
            idx = max(0,ac2rank_-1); lab.subplot(5, 10, 48);
            imx = denorm(loader.dataset[acsort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx); lab.axis('off'); lab.title(f'{idx}')
            #
            idx = ac2rank_; lab.subplot(5, 10, 49);
            lab.imshow(imc2); lab.axis('off'); lab.title(f'{idx}')
            #
            idx = min(cddts.size(0)-1, ac2rank_+2); lab.subplot(5, 10, 50)
            imx = denorm(loader.dataset[acsort[idx].item()][0]).numpy().transpose((1,2,0))
            lab.imshow(imx); lab.axis('off'); lab.title(f'{idx-1}')

            # dump
            lab.subplots_adjust(hspace=0.1, wspace=0.1)
            lab.savefig('x.svg', dpi=300)
            lab.show()


    @staticmethod
    def vis_caplus10(xr, r, imorig, # perturbed images, perturb, original images
            outorig, outadv, # original/adversarial distance matrix
            targetidx, # Q:idx for CA, C:idx for QA
            loader, # for loading images outside of the batch
            cddts,
            *,
            normimg=False, # for SOP
            metric=None,
            ):
        '''
        visualize CA+ effects, note, this is the w=2 version, i.e. we choose Q={q1,q2}
        '''
        assert(len(r[0].flatten()) == 224*224*3)
        assert(normimg == True)
        assert(len(targetidx[0]) == 10)
        SZ = (224, 224, 3)
        CMAP = None
        #lab.imshow(denorm(imorig[0]).numpy().transpose((1,2,0)))
        #lab.show()

        # traverse this batch of images.
        for i in range(xr.size(0)):
            # adversarial candidate
            imo = denorm(imorig[i]).detach().cpu().squeeze().numpy().transpose((1,2,0))
            ima = denorm(xr[i]).detach().cpu().squeeze().numpy().transpose((1,2,0))
            imr = xdnorm(r[i]).detach().cpu().squeeze().numpy().transpose((1,2,0))
            # idx of Q={...}
            tidx = targetidx[i]
            imqx, imqxv = [], []
            for j in range(10):
                imqj = denorm(loader.dataset[tidx[j].item()][0]).squeeze().numpy().transpose((1,2,0))
                imqx.append(imqj)
                imqjv = cddts[tidx[j]]
                imqxv.append(imqjv)
            qxcdist, qxcsort = [], []
            qxorank_, qxarank_ = [], []
            for j in range(10):
                # q_j and our candidate
                qjcdist = (cddts - imqxv[j].view(1, -1)).norm(2, dim=1)
                qxcdist.append(qjcdist)
                qjcsort = qjcdist.argsort()
                qxcsort.append(qjcsort)
                qjodist_ = (outorig[i] - imqxv[j]).norm(2)
                qjorank_ = (qjodist_ >= qxcdist[j]).sum()
                qxorank_.append(qjorank_)
                qjadist_ = (outadv[i] - imqxv[j]).norm(2)
                qjarank_ = (qjadist_ >= qxcdist[j]).sum()
                qxarank_.append(qjarank_)
            print('target (Q) idx = ', tidx)

            lab.figure(figsize=(21/3,64/3))
            # row 1: c, r, c+r
            lab.subplot(21,10, 2); lab.imshow(imo, cmap=CMAP); lab.title('$c$'); lab.axis('off')
            lab.subplot(21,10, 4); lab.imshow(imr, cmap=CMAP); lab.title('$r$'); lab.axis('off')
            lab.subplot(21,10, 6); lab.imshow(ima, cmap=CMAP); lab.title('$\\tilde{c}=c+r$'); lab.axis('off')

            # row 2,4,6,8,10,...: q_j, original
            for j in range(10):
                lab.subplot(21,10, 11+20*j); lab.imshow(imqx[j]); lab.title(f'$q_{{{j+1}}}$'); lab.axis('off')
                for k in range(5):
                    imx = denorm(loader.dataset[qxcsort[j][k].item()][0]).numpy().transpose((1,2,0))
                    lab.subplot(21,10, 12+20*j+k); lab.imshow(imx); lab.axis('off'); lab.title(f'{k}')
                # 8
                idx = max(0,qxorank_[j]-1); lab.subplot(21,10, 18+20*j);
                imx = denorm(loader.dataset[qxcsort[j][idx].item()][0]).numpy().transpose((1,2,0))
                lab.imshow(imx); lab.axis('off'); lab.title(f'{idx}')
                # 9, our c
                idx = qxorank_[j]; lab.subplot(21,10, 19+20*j);
                lab.imshow(imo); lab.axis('off'); lab.title(f'{idx}')
                # 10
                idx = min(cddts.size(0)-1, qxorank_[j]+1); lab.subplot(21,10, 20+20*j)
                imx = denorm(loader.dataset[qxcsort[j][idx].item()][0]).numpy().transpose((1,2,0))
                lab.imshow(imx); lab.axis('off'); lab.title(f'{idx}')

            ## row 3,5,7,9,11,...: q_j, adversarial candidate
            for j in range(10):
                lab.subplot(21,10, 21+20*j); lab.imshow(imqx[j]); lab.title(f'$q_{{{j+1}}}$'); lab.axis('off')
                ID = 0
                for k in range(5):
                    if qxarank_[j] == k:
                        imx = ima
                    else:
                        imx = denorm(loader.dataset[qxcsort[j][ID].item()][0]).numpy().transpose((1,2,0))
                        ID += 1
                    lab.subplot(21,10, 22+k+20*j); lab.imshow(imx); lab.axis('off'); lab.title(f'{k}')
                # 8
                idx = max(0, qxarank_[j]-1); lab.subplot(21,10, 28+20*j);
                imx = denorm(loader.dataset[qxcsort[j][idx].item()][0]).numpy().transpose((1,2,0))
                lab.imshow(imx); lab.axis('off'); lab.title(f'{idx}')
                # 9, our adversarial c~
                idx = qxarank_[j]; lab.subplot(21,10, 29+20*j);
                lab.imshow(ima); lab.axis('off'); lab.title(f'{idx}')
                # 10
                idx = qxarank_[j]; lab.subplot(21,10, 30+20*j)
                imx = denorm(loader.dataset[qxcsort[j][idx].item()][0]).numpy().transpose((1,2,0))
                lab.imshow(imx); lab.axis('off'); lab.title(f'{idx+1}')

            # typesetting, and dump
            lab.subplots_adjust(hspace=0.1, wspace=0.1)
            lab.savefig('x.svg', dpi=300)
            lab.show()
