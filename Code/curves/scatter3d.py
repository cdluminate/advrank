import sys, os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # noqa
import pylab as lab

# usage: {__file__} data
fig = lab.figure()
ax = fig.add_subplot(111, projection='3d')
N = 2000

X = np.loadtxt(sys.argv[1])
print(f'{sys.argv[1]} Shape:', X.shape)
ax.scatter(X[:N,0], X[:N,1], 1-X[:N,2], marker='o')
ax.scatter(X[:N,0], X[:N,1], 1-X[:N,3], marker='^')

ax.set_xlabel('DIST')
ax.set_ylabel('STD')
ax.set_zlabel('1-Rank%')

lab.show()
