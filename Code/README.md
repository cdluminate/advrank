Adversarial Attack and Defense for Deep Ranking
===============================================

The code will be released under an opensource license immediately after paper
acceptance, but currently it remains to be a confidential copy for peer review
only.

Usage
---

1. Download the corresponding datasets and adjust the path in the configuration
   file `config.yml`.

2. Train the baseline model (Cosine, Triplet, C2F2) on MNIST dataset using
   `python3 Train.py -M mnC_c2f2_trip`.

3. Attack the baseline model with "CA+" attack with 1 target queries with
   epsilon=0.3: `python3 Attack.py -M mnC_c2f2_trip -A C+:PGD-W1 -v -e 0.3`.

4. Adversarially train the baseline model with an epsilon=0.3 adversary:
   `python3 Train.py -M mnC_c2f2_trip -A 0.3`

5. Transfer attack, from resnet18 to c2f2:
   `python3 Attack.py -M mnC_res18_trip -A C+:PGD-W1 -v -e 0.3 -T mnC_c2f2_trip`

See also the content of `Makefile` for more detail.
The list of available attacks can be found in `Attack.py`. The list of
available models can be found in the following `Files` section.

Files
---

```
config.yml                    (configuration file: dataset paths, some experiment parameters)
Attack.py                     (script for adversarial ranking)
Train.py                      (script for baseline training and adversarial baseline training)
models/__init__.py            (python module)
models/common.py              (the common functions, including the attacking functions)
models/datasets/__init__.py   (python module)
models/datasets/fashion.py    (defines the fashion-mnist dataset reader)
models/datasets/mnist.py      (defines the mnist dataset reader)
models/datasets/sop.py        (defines the stanford online dataset reader)
models/dbg.py                 (utility functions for debugging)
models/faC_c2f2_siamese.py    (fashion-mnist, cosine, c2f2, siamese/contrastive)
models/faC_c2f2_trip.py       (fashion-mnist, cosine, c2f2, triplet)
models/faE_c2f2_siamese.py    (fashion-mnist, euclidean, c2f2, siamese)
models/faE_c2f2_trip.py       (fashion-mnist, euclidean, c2f2, triplet)
models/fa_cls_c2f2.py         (fashion-mnist, c2f2, classification)
models/fa_cls_softmax.py      (fashion-mnist, 1-layer softmax, classification)
models/mnC_c2f2_siamese.py    (mnist, c2f2, cosine, siamese)
models/mnC_c2f2_trip.py       (mnist, c2f2, cosine, triplet)
models/mnC_lenet_trip.py      (mnist, lenet, cosine, triplet)
models/mnC_res18_trip.py      (mnist, resnet18, cosine, triplet)
models/mnE_c2f2_siamese.py    (mnist, c2f2, euclidean, siamese)
models/mnE_c2f2_trip.py       (mnist, c2f2, euclidean, triplet)
models/mn_cls_c2f2.py         (mnist, c2f2, classification)
models/mn_cls_softmax.py      (mnist, c2f2, 1-layer softmax)
models/sopC_res18_siamese.py  (sop, resnet18, cosine, siamese)
models/sopC_res18_trip.py     (sop, resnet18, cosine, triplet)
models/sopE_res18_siamese.py  (sop, resnet18, euclidean, siamese)
models/sopE_res18_trip.py     (sop, resnet18, euclidean, triplet)
```

VersionInfo and Dependencies
---

We use the following software platform to run the experiments.

```
Platform: linux (Debian GNU/Linux Sid)
Python3: 3.7.3
Numpy: 1.16.2
Torch: 1.1.0 (CUDA 10)
Scipy: 1.2.1
Matplotlib: 3.0.3
```

Use the following command to install the missing python dependencies:
```
$ pip install -r requirements.txt
```

Copyright, License
---

```
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
```
