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
from . import mnC_c2f2_trip
import numpy as np

# reference: https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet18-mnist.ipynb


class BasicBlock(th.nn.Module):
    def __init__(self, inchan, outchan, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = th.nn.Conv2d(inchan, outchan, 3, stride=stride, padding=1, bias=False)
        self.bn1 = th.nn.BatchNorm2d(outchan)
        self.conv2 = th.nn.Conv2d(outchan, outchan, 3, stride=1, padding=1, bias=False)
        self.bn2 = th.nn.BatchNorm2d(outchan)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = th.nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is None:
            out = out + x
        else:
            out = out + self.downsample(x)
        return th.nn.functional.relu(out)


class ResNet(th.nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inchan = 64
        self.conv1 = th.nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = th.nn.BatchNorm2d(64)
        self.relu1 = th.nn.ReLU()
        self.maxpool = th.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.avgpool = th.nn.AvgPool2d(7, stride=1)
        #self.fc = th.nn.Linear(512, 10)
    def _make_layer(self, block, outchan, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = th.nn.Sequential(
                th.nn.Conv2d(self.inchan, outchan, 1, stride=stride, bias=False),
                th.nn.BatchNorm2d(outchan))
        layers = [block(self.inchan, outchan, stride, downsample)]
        self.inchan = outchan
        for i in range(1, blocks):
            layers.append(block(self.inchan, outchan))
        return th.nn.Sequential(*layers)
    def forward(self, x):
        # -1, 1, 28, 28
        x = self.conv1(x)
        # -1, 64, 14, 14
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        # -1, 64, 7, 7
        x = self.layer1(x)
        # -1, 64, 7, 7
        x = self.layer2(x)
        # -1, 128, 4, 4
        x = self.layer3(x)
        # -1, 256, 2, 2
        x = self.layer4(x)
        # -1, 512, 1, 1
        # because MNIST is already 1x1 here:
        # disable avg pooling
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def resnet18():
    return ResNet(block=BasicBlock, layers=[2,2,2,2])


class Model(mnC_c2f2_trip.Model):
    """
    LeNet-like convolutional neural network for embedding
    """
    def __init__(self):
        '''
        ResNet-18
        '''
        super(Model, self).__init__()
        self.setup_distance_metric()
        self.resnet18 = resnet18()

    def forward(self, x, *, l2norm=False):
        # -1, 1, 28, 28
        x = self.resnet18(x)
        # -1, 512
        if l2norm:
            x = x / x.norm(2, dim=1, keepdim=True).expand(*x.shape)
        return x
