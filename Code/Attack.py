#!/usr/bin/env python3
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
import sys, os, yaml, re
import numpy as np
import torch as th, torch.utils.data
import argparse, collections
from tqdm import tqdm
import models
from models.dbg import _bgcV, _fgcG


def Attack(argv):
    '''
    Attack a pre-trained model
    '''
    ag = argparse.ArgumentParser()
    ag.add_argument('-D', '--device', type=str,
            default='cuda' if th.cuda.is_available() else 'cpu')
    ag.add_argument('-A', '--attack', type=str, required=True,
            choices=[
                # untargeted attacks (pure)
                #; move the embedding off the original location and
                #; vastly change its semantics.
                'ES:FGSM-UT', 'ES:PGD-UT',

                # candidate attack (pure)
                #; only cares about absolute rank of target candidates
                'C+:PGD-W1', 'C+:PGD-W2', 'C+:PGD-W5', 'C+:PGD-W10',
                'C-:PGD-W1', 'C-:PGD-W2', 'C-:PGD-W5', 'C-:PGD-W10',

                # query attack (pure)
                #; only cares about absolute rank of target candidates
                'Q+:PGD-M1', 'Q+:PGD-M2', 'Q+:PGD-M5', 'Q+:PGD-M10',
                'Q-:PGD-M1', 'Q-:PGD-M2', 'Q-:PGD-M5', 'Q-:PGD-M10',

                # query attack + semantic preserving
                'SPQ+:PGD-M1', 'SPQ+:PGD-M2', 'SPQ+:PGD-M5', 'SPQ+:PGD-M10',
                'SPQ-:PGD-M1', 'SPQ-:PGD-M2', 'SPQ-:PGD-M5', 'SPQ-:PGD-M10',
                ])
    ag.add_argument('-e', '--epsilon', default=0.1,
            type=float, help='hyper-param epsilon | 1412.6572-FGSM min 0.007')
    ag.add_argument('-M', '--model', type=str, required=True)
    ag.add_argument('-T', '--transfer', type=str, required=False, default='')
    ag.add_argument('-v', '--verbose', action='store_true', help='verbose?')
    ag.add_argument('--vv', action='store_true', help='more verbose')
    ag = ag.parse_args(argv)

    print('>>> Parsing arguments and configuration file')
    for x in yaml.dump(vars(ag)).split('\n'): print(_fgcG(x))
    if ag.vv: ag.verbose = True
    config = yaml.load(open('config.yml', 'r').read(), Loader=yaml.SafeLoader)
    print(_bgcV('Attacking method is', ag.attack, f'\u03b5={ag.epsilon}'))

    # Load the white-box attacking target model
    if re.match('\S+:\S+', ag.model):
        Mname, Mpath = re.match('(\S+):(\S+)', ag.model).groups()
    else:
        Mname, Mpath = ag.model, 'trained/' + ag.model + '.sdth'
    print(f'>>> Loading white-box target {Mname} model from:', Mpath)
    model = getattr(models, Mname).Model().to(ag.device)
    model.load_state_dict(th.load(Mpath))
    print(model)

    if ag.transfer:
        if re.match('\S+:\S+', ag.transfer):
            # also specified the path of the blackbox model
            Tname, Tpath = re.match('(\S+):(\S+)', ag.transfer).groups()
        else:
            # load it from the default path
            Tname, Tpath = ag.transfer, 'trained/'+ag.transfer+'.sdth'
        print(f'>>> Loading {Tname} from {Tpath} for blackbox/transfer attack.')
        modelT = getattr(models, Tname).Model().to(ag.device)
        modelT.load_state_dict(th.load(Tpath))
        modelT.eval()

    print('>>> Loading dataset ...', end=' ')
    if not ag.vv:
        loader_test = \
            model.getloader('test', config[Mname]['batchsize_atk'])
        if ag.attack in ('Q:PGD-M1000', 'Q:PGD-M10000', 'Q:PGD-MX'):
            loader_test = model.getloader('test', 10) # override batchsize
    elif ag.vv:
        loader_test = model.getloader('test', 1)
        print('| overriden batchsize to 1', end=' ')
    print('| Testing dataset size =', len(loader_test.dataset))

    print('>>> Start Attacking ...')
    dconf = {'epsilon': ag.epsilon}
    if ag.transfer:
        dconf['TRANSFER'] = {'transfer': Tname, 'model': modelT, 'device': ag.device}
    model.attack(ag.attack, loader_test, dconf=dconf, verbose=ag.verbose)


if __name__ == '__main__':
    Attack(sys.argv[1:])
