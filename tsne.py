import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.cluster import SpectralClustering 
from sklearn.cluster import DBSCAN 
import sys
import matplotlib.pyplot as plt
import datetime
from minisom import MiniSom

import csv
import os

DIR = 'stanton-csv'

X = []

ctr = 0
for filename in os.listdir(DIR):
    #print("{}: {}".format(str(ctr).zfill(4), filename))
    ctr += 1

    if ctr > 1000:
        break

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

def do_nothing(X):
    return X

TSNE_CACHE = None
def do_tsne(X):
    global TSNE_CACHE
    if TSNE_CACHE is None:
        TSNE_CACHE = TSNE(n_components=2).fit_transform(X)
    return TSNE_CACHE

def do_pca(X):
    return PCA(n_components=2).fit_transform(X)

def do_pca_radial(X):
    return KernelPCA(n_components=2, kernel="rbf").fit_transform(X)

def do_som(X):
    inp_len = X.shape[1]
    print("inp_len = {}".format(inp_len))
    som = MiniSom(100, 100, inp_len, sigma=0.3, learning_rate=5)
    som.train_random(X, 10)
    X_out = []
    for x in X:
        w = som.winner(x)
        X_out.append(som.winner(x))
    X_out = np.array(X_out)
    return X_out

def do_dbscan(eps, min_samples, X):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_embedded)
    return db.labels_

def do_spectral_clustering(X):
    return SpectralClustering(
            n_clusters=8,
            affinity='rbf',
            assign_labels='kmeans'
    ).fit_predict(X)

reduction_methods = [
        ("plain_tsne_post", do_nothing, do_tsne),
        ("tsne", do_tsne, do_nothing)
        #("som", do_som, do_nothing)
        #("pca", do_pca),
        #("pca_radial", do_pca_radial)
]

e = 7
m = 50
cluster_methods = [
    ("DBSCAN_eps=4_min=10",  lambda X: do_dbscan(4, 10, X))
    #("SpCl", do_spectral_clustering)
]
fmt = 'pdf'
#fmt = 'png'

#COLORS = 'rbcmyg'
COLORS = [
    'xkcd:purple',
    'xkcd:green',
    'xkcd:blue',
    'xkcd:pink',
    'xkcd:red',
    'xkcd:teal',
    'xkcd:orange',
    'xkcd:cyan',
    'xkcd:yellow',
    'xkcd:light green',
    'xkcd:dark green',
    'xkcd:navy'
]
def label2color(l):
    if l >= 0:
        if l >= len(COLORS):
            print("Too many labels")
            sys.exit(1)
        return COLORS[l%len(COLORS)]
    else:
        return 'k'

for (rname,rf,post) in reduction_methods:
    print("doing {}".format(rname))
    X_embedded = rf(np.array(X))
    for (cname,cf) in cluster_methods:
        print("    doing {}".format(cname))
        fig = plt.figure()
        fig.suptitle("{} {}".format(rname, cname))
        ax = fig.add_subplot(111)
        db = DBSCAN(eps=4, min_samples=10).fit(X_embedded)
        labels = cf(X_embedded)
        label_indxs = {}
        for (i,l) in enumerate(labels):
            if l in label_indxs:
                label_indxs[l].append(i)
            else:
                label_indxs[l] = [i]
        print("doing post proc")
        X_embedded = post(X_embedded)
        for l in label_indxs:
            print("scattering label {}".format(l))
            idxs = label_indxs[l]
            color = label2color(l)
            ax.scatter(X_embedded[idxs,0],X_embedded[idxs,1], c=color)
        plt.savefig("outs/{}-{}-{}.{}".format(
            rname, cname,
            datetime.datetime.now().timestamp(),
            fmt
        ))
        ax.plot()

#plt.plot(X_embedded[:,0], X_embedded[:,1], 'b.')
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X_embedded[:,0],X_embedded[:,1],X_embedded[:,2], c='b')
#plt.savefig("outs/tsne-{}.pdf".format(datetime.datetime.now().timestamp()))
plt.show()

