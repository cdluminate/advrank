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
# This file imports models that can be specified by "Train.py -M"
# Note, common abbr: C(osine) E(uclidean)

# fashion-mnist (fa)
from . import fa_cls_softmax
from . import fa_cls_c2f2

from . import faC_c2f2_siamese
from . import faC_c2f2_trip
from . import faE_c2f2_siamese
from . import faE_c2f2_trip

from . import faC_lenet_trip
from . import faC_res18_trip

# mnist (mn) (inherit)-> fashion-mnist
from . import mn_cls_softmax
from . import mn_cls_c2f2

from . import mnC_c2f2_siamese
from . import mnC_c2f2_trip
from . import mnE_c2f2_siamese
from . import mnE_c2f2_trip

from . import mnC_lenet_trip
from . import mnC_res18_trip

# stanfard online product (sop)
from . import sopC_res18_siamese
from . import sopC_res18_trip
from . import sopE_res18_siamese
from . import sopE_res18_trip
