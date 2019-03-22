import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.cluster import spectral_clustering 
import sys
import matplotlib.pyplot as plt
import datetime

import csv
import os

DIR = 'stanton-csv'

X = []

ctr = 0
for filename in os.listdir(DIR):
    #print("{}: {}".format(str(ctr).zfill(4), filename))
    ctr += 1

    #if ctr > 30:
    #    break

    amplis = []
    path = '{}/{}'.format(DIR, filename)
    reader = csv.reader(open(path, 'r'))

    bad_ampl = False
    for row in reader:
        angle = float(row[0].strip())
        ampli = float(row[1].strip())
        if angle > 45.0 or angle < -45.0:
            continue
        if ampli < 0.0001:
            bad_ampl = True
            break
        amplis.append(ampli)

    if not bad_ampl:
        point = np.array(amplis)
        #X = np.append(X, point, axis=0)
        X.append(point)


def do_tsne(X):
    return TSNE(n_components=2).fit_transform(X)

def do_pca(X):
    return PCA(n_components=2).fit_transform(X)

def do_kernel_pca(X):
    return KernelPCA(n_components=2, kernel="poly", gamma=5).fit_transform(X)

methods = [
        #("tsne", do_tsne),
        #("pca", do_pca),
        ("kernelPCA", do_kernel_pca)
]

for (name,f) in methods:
    print("Doing {}".format(name))
    fig = plt.figure()
    fig.suptitle(name)
    ax = fig.add_subplot(111)
    X_embedded = f(X)
    ax.scatter(X_embedded[:,0],X_embedded[:,1], c='b')
    plt.savefig("outs/{}-{}.pdf".format(name,
                                        datetime.datetime.now().timestamp()))
    ax.plot()

plt.plot(X_embedded[:,0], X_embedded[:,1], 'b.')
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X_embedded[:,0],X_embedded[:,1],X_embedded[:,2], c='b')
#plt.savefig("outs/tsne-{}.pdf".format(datetime.datetime.now().timestamp()))
plt.show()

