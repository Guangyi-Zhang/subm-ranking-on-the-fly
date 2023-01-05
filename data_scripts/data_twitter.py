import numpy as np
from numpy.linalg import norm
import pandas as pd
import pickle
from collections import defaultdict


rng = np.random.default_rng(12345)
fdata = '/NOBACKUP/guaz/msr-datasets/glove.twitter.27B.25d.txt'


def sim(a,b):
    return np.dot(a, b) / (norm(a)*norm(b))


def similar_to(kw, nmax):
    w2v = dict()
    tar = None
    with open(fdata, 'r') as fin:
        for line in fin:
            els = line.split(' ')
            w, v = els[0], els[1:]
            v = [float(i) for i in v]
            w2v[w] = v

            if w == kw:
                tar = v

    ws = [(w,sim(tar, v)) for w,v in w2v.items()]
    ws = sorted(ws, key=lambda x: x[1])

    ws = ws[-nmax:]
    W = [w for w,s in ws]
    V = np.array([w2v[w] for w,s in ws])
    print(V.shape)
    for w,_ in reversed(ws[-50:]):
        print(w)

    pickle.dump((W,V,tar), open(f'datasets/synonym-{kw}.pkl', 'wb'))


if __name__ == '__main__':
    kw = 'trump'
    similar_to('trump', nmax=100000)
