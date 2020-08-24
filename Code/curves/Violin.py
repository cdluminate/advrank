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
import seaborn as sns
import numpy as np
import torch as th
import pylab as lab
import pandas as pd
import matplotlib.pyplot as plt



Vsd = th.load('./pretrained/trained.vanilla/mnC_c2f2_trip.sdth')
Dsd = th.load('./pretrained/trained.mintmaxe/mnC_c2f2_trip.sdth')
#odict_keys(['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fc1.weight', 'fc1.bias'])

#print('param', 'shape', 'mean', 'min', 'max', sep='\t')


# conv1.w
df1 = pd.DataFrame({'index': 0,
    'weight': Vsd['conv1.weight'].numpy().ravel(),
    'defense': False,
    'layer': 'conv1.weight'})
#print('[V] conv1.W', x.shape, x.mean().item(), x.min().item(), x.max().item(), sep='\t')
df2 = pd.DataFrame({'index': 0,
    'weight': Dsd['conv1.weight'].numpy().ravel(),
    'defense': True,
    'layer': 'conv1.weight'})
#print('[D] conv1.W', x.shape, x.mean().item(), x.min().item(), x.max().item(), sep='\t')
df = pd.concat((df1, df2))
sns.set(style="whitegrid", palette="pastel", color_codes=True)
pl = sns.violinplot(data=df, y='weight', x='defense', split=True, inner='quart')
pl.set_xlabel('With our defense?')
pl.set_title('Weights of Conv1')
lab.savefig('curves/vlconv1w.svg')
plt.clf()
print('conv1.w ok')

# conv1.b
df1 = pd.DataFrame({'index': 0,
    'bias': Vsd['conv1.bias'].numpy().ravel(),
    'defense': False,
    'layer': 'conv1.bias'})
#print('[V] conv1.W', x.shape, x.mean().item(), x.min().item(), x.max().item(), sep='\t')
df2 = pd.DataFrame({'index': 0,
    'bias': Dsd['conv1.bias'].numpy().ravel(),
    'defense': True,
    'layer': 'conv1.bias'})
#print('[D] conv1.W', x.shape, x.mean().item(), x.min().item(), x.max().item(), sep='\t')
df = pd.concat((df1, df2))
sns.set(style="whitegrid", palette="pastel", color_codes=True)
pl = sns.violinplot(data=df, y='bias', x='defense', split=True, inner='quart')
pl.set_xlabel('With our defense?')
pl.set_title('Bias of Conv1')
lab.savefig('curves/vlconv1b.svg')
plt.clf()
print('conv1.b ok')


# conv2.w
df1 = pd.DataFrame({'index': 0,
    'weight': Vsd['conv2.weight'].numpy().ravel(),
    'defense': False,
    'layer': 'conv2.weight'})
df2 = pd.DataFrame({'index': 0,
    'weight': Dsd['conv2.weight'].numpy().ravel(),
    'defense': True,
    'layer': 'conv2.weight'})
df = pd.concat((df1, df2))
sns.set(style="whitegrid", palette="pastel", color_codes=True)
pl = sns.violinplot(data=df, y='weight', x='defense', split=True, inner='quart')
pl.set_xlabel('With our defense?')
pl.set_title('Weights of Conv2')
lab.savefig('curves/vlconv2w.svg')
plt.clf()
print('conv2.w ok')

# conv2.b
df1 = pd.DataFrame({'index': 0,
    'bias': Vsd['conv2.bias'].numpy().ravel(),
    'defense': False,
    'layer': 'conv2.bias'})
df2 = pd.DataFrame({'index': 0,
    'bias': Dsd['conv2.bias'].numpy().ravel(),
    'defense': True,
    'layer': 'conv2.bias'})
df = pd.concat((df1, df2))
sns.set(style="whitegrid", palette="pastel", color_codes=True)
pl = sns.violinplot(data=df, y='bias', x='defense', split=True, inner='quart')
pl.set_xlabel('With our defense?')
pl.set_title('Bias of Conv2')
lab.savefig('curves/vlconv2b.svg')
plt.clf()
print('conv2.b ok')

# fc1 w
df1 = pd.DataFrame({'index': 0,
    'weight': Vsd['fc1.weight'].numpy().ravel(),
    'defense': False,
    'layer': 'fc1.weight'})
df2 = pd.DataFrame({'index': 0,
    'weight': Dsd['fc1.weight'].numpy().ravel(),
    'defense': True,
    'layer': 'fc1.weight'})
df = pd.concat((df1, df2))
sns.set(style="whitegrid", palette="pastel", color_codes=True)
pl = sns.violinplot(data=df, y='weight', x='defense', split=True, inner='quart')
pl.set_xlabel('With our defense?')
pl.set_title('Weights of FC1')
lab.savefig('curves/vlfc1w.svg')
plt.clf()
print('fc1.w ok')

# fc1 b
df1 = pd.DataFrame({'index': 0,
    'bias': Vsd['fc1.bias'].numpy().ravel(),
    'defense': False,
    'layer': 'fc1.bias'})
df2 = pd.DataFrame({'index': 0,
    'bias': Dsd['fc1.bias'].numpy().ravel(),
    'defense': True,
    'layer': 'fc1.bias'})
df = pd.concat((df1, df2))
sns.set(style="whitegrid", palette="pastel", color_codes=True)
pl = sns.violinplot(data=df, y='bias', x='defense', split=True, inner='quart')
pl.set_xlabel('With our defense?')
pl.set_title('Bias of FC1')
lab.savefig('curves/vlfc1b.svg')
plt.clf()
print('fc1.b ok')

#x = Vsd['conv2.weight']
#df3 = pd.DataFrame({'index': 0, 'weight': x.numpy().ravel(), 'defense': False, 'layer': 'conv2'})
#x = Dsd['conv2.weight']
#df4 = pd.DataFrame({'index': 0, 'weight': x.numpy().ravel(), 'defense': True, 'layer': 'conv2'})
#
#x = Vsd['fc1.weight']
#df5 = pd.DataFrame({'index': 0, 'weight': x.numpy().ravel(), 'defense': False, 'layer': 'fc1'})
#x = Dsd['fc1.weight']
#df6 = pd.DataFrame({'index': 0, 'weight': x.numpy().ravel(), 'defense': True, 'layer': 'fc1'})
#
#df = pd.concat([df1, df2, df3, df4, df5, df6])
#print(df)
## sns.violinplot(x='day', y='total_bill', hue='smoker', split=True, inner='quart', data=tips, palette='pastel')
#sns.set(style="whitegrid", palette="pastel", color_codes=True)
#sns.violinplot(data=df, y='weight', x='layer', hue='defense', split=True, inner='quart', palette={True: "y", False: "b"})
##lab.show()
#lab.savefig('curves/violin.svg')

