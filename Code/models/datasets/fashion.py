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
import gzip
import numpy as np
import torch as th
import torch.utils.data


def get_dataset(path: str, kind='train'):
    """
    Load MNIST data from `path`
    """
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def get_loader(path: str, batchsize: int, kind='train'):
    """
    Load MNIST data and turn them into dataloaders
    """
    if kind == 'train':
        x_train, y_train = get_dataset(path, kind='train')
        x_train = th.from_numpy(x_train).float().view(-1, 1, 28, 28) / 255.
        y_train = th.from_numpy(y_train).long().view(-1, 1)
        data_train = th.utils.data.TensorDataset(x_train, y_train)
        loader_train = th.utils.data.DataLoader(data_train,
                                                batch_size=batchsize, shuffle=True,
                                                pin_memory=True)
        return loader_train
    else:
        x_test, y_test = get_dataset(path, kind='t10k')
        x_test = th.from_numpy(x_test).float().view(-1, 1, 28, 28) / 255.
        y_test = th.from_numpy(y_test).long().view(-1, 1)
        data_test = th.utils.data.TensorDataset(x_test, y_test)
        loader_test = th.utils.data.DataLoader(data_test,
                                               batch_size=batchsize, shuffle=False,
                                               pin_memory=True)
        return loader_test


def get_label(n):
    '''
    Get the label list
    '''
    return """
    T-shirt/top
    Trouser
    Pullover
    Dress
    Coat
    Sandal
    Shirt
    Sneaker
    Bag
    Ankle boot
    """.split()[n]
