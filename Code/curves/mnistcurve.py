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
import numpy as np
import pylab as lab
import matplotlib.pyplot as plt

str2flist = lambda x: [float(y) for y in x.split()]

eps = [x*0.01 for x in range(30+1)]

# the following numbers are automatically collected from batch experiment

es_o=str2flist('''
-0.000
0.003
0.011
0.026
0.048
0.077
0.113
0.157
0.210
0.270
0.340
0.421
0.512
0.611
0.719
0.834
0.919
0.996
1.064
1.124
1.176
1.220
1.258
1.290
1.317
1.339
1.357
1.372
1.383
1.391
1.395
''')

cp_o=str2flist('''
0.499
0.441
0.389
0.333
0.291
0.252
0.220
0.190
0.168
0.145
0.127
0.113
0.098
0.087
0.076
0.068
0.060
0.054
0.049
0.043
0.039
0.035
0.032
0.030
0.027
0.026
0.024
0.023
0.022
0.021
0.020
''')

cm_o=str2flist('''
0.021
0.034
0.048
0.062
0.075
0.085
0.094
0.103
0.114
0.129
0.153
0.190
0.243
0.313
0.401
0.500
0.578
0.645
0.705
0.753
0.792
0.826
0.851
0.874
0.888
0.903
0.913
0.921
0.928
0.933
0.937
''')

qp_o=str2flist('''
0.503
0.447
0.397
0.354
0.315
0.274
0.239
0.211
0.185
0.164
0.143
0.127
0.110
0.099
0.086
0.083
0.075
0.074
0.070
0.070
0.066
0.066
0.064
0.065
0.064
0.064
0.064
0.064
0.063
0.063
0.064
''')

qm_o=str2flist('''
0.005
0.009
0.013
0.019
0.025
0.030
0.036
0.042
0.046
0.051
0.057
0.059
0.067
0.068
0.075
0.074
0.080
0.079
0.083
0.082
0.085
0.084
0.087
0.086
0.086
0.086
0.087
0.087
0.088
0.088
0.087
''')

es_a=str2flist('''
-0.000
0.000
0.000
0.001
0.001
0.001
0.002
0.003
0.003
0.004
0.005
0.006
0.007
0.008
0.010
0.011
0.013
0.014
0.016
0.017
0.019
0.020
0.022
0.024
0.026
0.027
0.029
0.031
0.032
0.034
0.035
''')

cp_a=str2flist('''
0.497
0.490
0.485
0.471
0.468
0.460
0.451
0.442
0.434
0.429
0.421
0.413
0.406
0.399
0.391
0.386
0.380
0.370
0.372
0.363
0.353
0.348
0.345
0.338
0.335
0.324
0.324
0.322
0.316
0.312
0.306
''')

cm_a=str2flist('''
0.020
0.022
0.023
0.025
0.027
0.028
0.031
0.032
0.034
0.036
0.038
0.040
0.042
0.044
0.046
0.048
0.050
0.051
0.053
0.055
0.057
0.058
0.059
0.061
0.062
0.064
0.065
0.066
0.067
0.069
0.070
''')

qp_a=str2flist('''
0.498
0.496
0.490
0.482
0.473
0.468
0.464
0.453
0.445
0.445
0.435
0.434
0.420
0.414
0.405
0.405
0.393
0.392
0.384
0.378
0.376
0.373
0.366
0.364
0.360
0.354
0.351
0.347
0.340
0.334
0.332
''')

qm_a=str2flist('''
0.005
0.005
0.006
0.006
0.007
0.007
0.007
0.008
0.008
0.009
0.010
0.010
0.011
0.012
0.012
0.013
0.013
0.014
0.015
0.016
0.017
0.017
0.018
0.019
0.020
0.020
0.021
0.022
0.022
0.023
0.024
''')

_s = lambda x: [y*100 for y in x]

fig = lab.figure(figsize=[12,5])

lp = fig.add_subplot(121)
lp.grid(linestyle='--')
cp = lp.plot(eps, _s(cp_o), marker='^', color='tab:red')
cm = lp.plot(eps, _s(cm_o), marker='v', color='tab:blue')
qp = lp.plot(eps, _s(qp_o), marker='2', color='tab:orange')
qm = lp.plot(eps, _s(qm_o), marker='1', color='tab:cyan')
lp.set_ylim(0, 100)
lp.set_xlabel('$\epsilon$')
lp.set_ylabel('Rank (normalized)')
lp.set_title('Attacking Vanilla Model')
lp.legend([
    'CA+ ($w=1$)',
    'CA- ($w=1$)',
    'SP-QA+ ($m=1$)',
    'SP-QA- ($m=1$)'], loc='upper left')
ax2 = lp.twinx()
ax2.set_ylim(0, 2)
ax2.plot(eps, es_o, marker='.', color='grey')
ax2.legend(['Distance'], loc=7)
ax2.set_xlabel('$\epsilon$')
ax2.set_ylabel('Max Shift Distance ($\in [0,2]$)')
#plt.tight_layout(h_pad=5.0)

rp = fig.add_subplot(122)
rp.grid(linestyle='--')
#rp.plot(eps, es_a, marker='o')
rp.plot(eps, _s(cp_a), marker='^', color='tab:red')
rp.plot(eps, _s(cm_a), marker='v', color='tab:blue')
rp.plot(eps, _s(qp_a), marker='2', color='tab:orange')
rp.plot(eps, _s(qm_a), marker='1', color='tab:cyan')
rp.set_ylim(0, 100)
rp.set_xlabel('$\epsilon$')
rp.set_ylabel('Rank (normalized)')
rp.set_title('Attacking Defensive Model')
rp.legend([
    'CA+ ($w=1$)',
    'CA- ($w=1$)',
    'SP-QA+ ($m=1$)',
    'SP-QA- ($m=1$)'], loc='upper left')
ax2 = rp.twinx()
ax2.set_ylim(0, 2)
ax2.plot(eps, es_a, marker='.', color='grey')
ax2.legend(['Distance'], loc=7)
ax2.set_xlabel('$\epsilon$')
ax2.set_ylabel('Max Shift Distance ($\in [0,2]$)')
plt.tight_layout(h_pad=10.0)

#lab.show()
lab.savefig(f'{__file__}.svg')

