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
import torchvision as vision
from PIL import Image
import random
from collections import defaultdict


def getTransform(kind: str):
    '''
    training: (orig) -> resize (256,256) -> randcrop (224,224)
    testing: (orig) -> resize (256,256) -> centercrop (224,224)
    '''
    transforms = []
    normalizer = vision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if kind == 'train':
        transforms.append(vision.transforms.Resize((256, 256)))
        transforms.append(vision.transforms.RandomHorizontalFlip(p=0.5))
        transforms.append(vision.transforms.RandomCrop((224, 224)))
        transforms.append(vision.transforms.ToTensor())
        transforms.append(normalizer)
    elif kind == 'test':
        transforms.append(vision.transforms.Resize((256, 256)))
        transforms.append(vision.transforms.CenterCrop((224, 224)))
        transforms.append(vision.transforms.ToTensor())
        transforms.append(normalizer)
    else:
        raise ValueError(f'what is {kind} dataset?')
    return vision.transforms.Compose(transforms)


class SOPDataset(th.utils.data.Dataset):
    '''
    For loading a single image from the dataset.
    '''

    def __init__(self, basepath, listfile):
        self.basepath = basepath
        self.listfile = listfile
        self.kind = 'train' if 'train' in listfile else 'test'
        self.transform = getTransform(self.kind)
        with open(os.path.join(basepath, listfile), 'rt') as fp:
            lines = [x.split() for x in fp.readlines()[1:]]
        print(f'SOPDataset[{self.kind}]: Got {len(lines)} Images.')

        # build helper mappings
        self.ImageId2Path = {int(x[0]): str(x[3]) for x in lines}
        self.ImageId2Class = {int(x[0]): int(x[1]) for x in lines}
        self.Class2ImageIds = defaultdict(list)
        for x in lines:
            self.Class2ImageIds[int(x[1])].append(int(x[0]))

    def __len__(self):
        return len(self.ImageId2Path)

    def __getitem__(self, index: int, *, byiid=False):
        '''
        Get image by ImageID
        '''
        if byiid == False and index >= len(self):
            raise IndexError
        if byiid and index not in self.ImageId2Path:
            raise IndexError
        idx_offset = 1 if self.kind == 'train' else 59552
        if byiid:
            idx_offset = 0
        path = str(self.ImageId2Path[index + idx_offset])
        label = int(self.ImageId2Class[index + idx_offset])
        impath = os.path.join(self.basepath, path)
        image = Image.open(impath).convert('RGB')
        image = self.transform(image)
        return image, label


class SOPSiameseDataset(th.utils.data.Dataset):
    '''
    Produce Datapairs for Siamese/Contrastive
    '''

    def __init__(self, basepath, listfile):
        self.data = SOPDataset(basepath, listfile)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError
        anchor_im, anchor_label = self.data[index]
        iids = [index]
        label_same = []
        if random.random() > 0.5:  # positive one
            iid = random.choice(self.data.Class2ImageIds[anchor_label])
            iids.append(iid)
            another_im, another_label = self.data.__getitem__(iid, byiid=True)
            label_same.append(1)
        else:  # negative one
            lid = random.randint(0, len(self)-1)
            iids.append(lid)
            another_im, another_label = self.data[lid]
            issame = 0 if another_label != anchor_label else 1
            label_same.append(issame)
        images = th.stack([anchor_im, another_im])
        #iids = th.LongTensor(iids)
        label = th.LongTensor(label_same)

        return images, label


class SOPTripletDataset(th.utils.data.Dataset):
    '''
    Produce Datapairs for Triplet
    '''

    def __init__(self, basepath, listfile):
        self.data = SOPDataset(basepath, listfile)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError
        anchor_im, anchor_label = self.data[index]
        # positive
        pid = random.choice(self.data.Class2ImageIds[anchor_label])
        positive_im, positive_label = self.data.__getitem__(pid, byiid=True)
        # netgative
        nid = random.randint(0, len(self)-1)
        negative_im, negative_label = self.data[nid]
        iids = th.LongTensor([index, pid, nid])
        images = th.stack([anchor_im, positive_im, negative_im])
        return images, iids


def get_dataset(path: str, ntuple: int, split='train'):
    '''
    Load Stanford Online Products Dataset
    '''
    basepath = path
    listfile = 'Ebay_train.txt' if split == 'train' else 'Ebay_test.txt'
    if split == 'train':
        datasetclass = SOPSiameseDataset if ntuple == 2 else SOPTripletDataset
    elif split == 'test':
        datasetclass = SOPDataset
        if ntuple != 1:
            raise ValueError('ntuple > 1 for testing set???')
    else:
        raise NotImplementedError
    return datasetclass(basepath, listfile)


def get_loader(path: str, batchsize: int, ntuple: int, split: str):
    """
    Load SOP dataset and turn them into dataloaders
    """
    dataset = get_dataset(path, ntuple, split)
    if split == 'train':
        loader = th.utils.data.DataLoader(dataset, batch_size=batchsize,
                                          shuffle=True, pin_memory=True, num_workers=8,
                                          )
        return loader
    else:
        loader = th.utils.data.DataLoader(dataset, batch_size=batchsize,
                                          shuffle=False, pin_memory=False, num_workers=4)
        return loader
