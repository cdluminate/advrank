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
from . import sopC_res18_trip

class Model(sopC_res18_trip.Model):
    """
    Res18 + SOP dataset, Triplet Loss function
    """
    def setup_distance_metric(self, metric='E'):
        self.metric = metric

__performance='''
When setting FINETUNE=1 + lock param (default)
 reported in paper
 ES -> ?
 CA+ (w=1) -> adv=0.0(/100)
 CA- (w=1) -> adv=100.0(/100)
 QA+ (m=1) -> adv=0.8(/100)
 QA- (m=1) -> adv=77.7(/100)

When setting FINETUNE=1 + lock + adv training (default defense)
 reported in paper
 ES -> ?
 CA+ (w=1) -> adv=0.0(/100)
 CA- (w=1) -> adv=100.0(/100)
 QA+ (m=1) -> adv=5.0(/100)
 QA- (m=1) -> adv=59.8(/100)

When setting FINETUNE=1 + unlock
 42 eph --> r@mean 133.9/60502, r@1 65.3/100, r@10 81.7/100, r@100 92.0/100
 [adv attack]
 path: ../pretrained/trained.extra/preunlock/sopE_res18_trip.sdth
 ES:PGD-UT -> baseline=65.285/100, adv=0.033/100, embshift=60.045
 CA+ (w=1) -> baseline=0.500(/1), adv=0.000(/1), embshift=16.665
 CA- (w=1) -> baseline=0.017(/1), adv=1.000(/1), embshift=19.849
 QA+ (m=1) -> baseline=0.501(/1), adv=0.000(/1), embshift=15.710
 QA- (m=1) -> baseline=0.005(/1), adv=0.999(/1), embshift=28.919

When setting FINETUNE=1 + unlock + adv training
 42 eph --> r@mean 2222/60502, r@1 25.5/100, r@10 37.5/100, r@100 53.2/100
 [adv attack]
 path: ../pretrained/trained.extra/preunlock+adv/sopE_res18_trip.sdth
 ES:PGD-UT -> baseline=25.564/100, adv=0.402/100, embshift=3.613
 CA+ (w=1) -> baseline=0.500(/1), adv=0.109(/1), embshift=2.878
 CA- (w=1) -> baseline=0.011(/1), adv=0.507(/1), embshift=3.081
 QA+ (m=1) -> baseline=0.500(/1), adv=0.135(/1), embshift=2.861
 QA- (m=1) -> baseline=0.005(/1), adv=0.480(/1), embshift=3.327

When setting FINETUNE=0 (unlock)
 42 eph --> R@1 45.3, R@10 62.0, R@100 77.6 (/100)
 continue training with 0.1*lr only results in slight improvement.
 [adv attack]
 path: ../pretrained/trained.extra/nopre/sopE_res18_trip.sdth
 ES:PGD-UT -> baseline=45.144/100, adv=0.008/100 embshift=33.912

When setting FINETUNE=0 (unlock) + Adversarial training
 42 eph --> R@1 25.4, R@10 37.2, R@100 53.3 (/100)
 continue training makes nearly no difference.
 [adv attack]
 path: ../pretrained/trained.extra/nopre+adv/sopE_res18_trip.sdth
 ES:PGD-UT -> baseline=25.414/100, adv=0.499/100, embshift=3.005
 CA+ (w=1) -> baseline=0.499(/1), adv=0.097(/1), embshift=2.516
 CA- (w=1) -> baseline=0.011(/1), adv=0.502(/1), embshift=2.684
 QA+ (m=1) -> baseline=0.501(/1), adv=0.128(/1), embshift=2.458
 QA- (m=1) -> baseline=0.005(/1), adv=0.492(/1), embshift=2.924
'''
