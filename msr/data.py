import numpy as np
from numpy.linalg import norm
import pickle
from sklearn.model_selection import train_test_split
from collections import defaultdict
from functools import partialmethod, partial

from msr.base import Seq, Data, Stream


class DemandStream(Stream):
    def __init__(self, demands, T: int=None):
        super().__init__()
        self.F = demands
        self.T = len(self.F)*2 if T is None else T # horizon

    def __iter__(self):
        ''' yield [(f, k),...] '''
        arrivals = np.random.randint(0, self.T, size=len(self.F)) # random arrivals
        t2f = defaultdict(list)
        for t,fk in zip(arrivals, self.F):
            t2f[t].append(fk)
        for t in range(self.T + 100): # avoid trunating last f
            yield t2f[t]


class ItemStream(Stream):
    def __init__(self, items):
        super().__init__()
        self.V = items

    def __iter__(self):
        n = len(self.V)
        arrivals = np.random.permutation(n)
        for i in arrivals:
            yield self.V[i]


class CoverageData(Data):
    '''
    f as a coverage function.
    In a network, a node takes its neighborhood as coverage.
    '''
    def __init__(self, V=None, g=None):
        '''
        g: a cardinality-based concave function, e.g., g_T(S) = sqrt(|S cap T|) / sqrt(|T|)
        '''
        super().__init__()
        self.V = V
        self.n = None if V is None else len(V)
        self.V2id = dict()
        if V is not None:
            for i,v in enumerate(V):
                self.V2id[v] = i
        self.Fraw = list()
        self.g = g

    def S2V(self, S):
        return S

    def makeF(self):
        # Turn every subset in F_raw into a func f
        def _f(S, i):
            if len(S) == 0:
                return 0
            S = set(S)
            Sf,_ = self.Fraw[i]
            R, N1, renew = self.cache(S, i)
            if R is not None: # cache hit
                if len(R) == 0:
                    return len(N1) / len(Sf)
                #taken = self.S2V(R).union(N1)
                taken = self.S2V(R)
                taken = taken.intersection(Sf)
                taken = taken.union(N1)
            else:
                taken = self.S2V(S)
                taken = taken.intersection(Sf)

            if renew:
                self.caches[i] = (S,taken) # renew cache
            return len(taken) / len(Sf)

        for i, (s,k) in enumerate(self.Fraw):
            self.caches.append((set(),set())) # (last S, N(S))
            self.F.append((partial(_f, i=i), k))

    def next_demand(self, s, k: int):
        '''
        s: a subset
        k: cardinality
        '''
        if s is None:
            if self.n is None:
                self.n = len(self.V2id)
                self.V = np.arange(self.n)
            self.makeF()
            return

        news = []
        for v in s:
            if self.n is None and v not in self.V2id:
                self.V2id[v] = len(self.V2id)
            news.append(self.V2id[v])
        self.Fraw.append((set(news),k))


class NetworkData(CoverageData):
    '''
    f as a coverage function.
    In a network, a node takes its neighborhood as coverage.
    '''
    def __init__(self, N: dict, V=None):
        '''
        N: neighbood node-to-set
        '''
        super().__init__(V=V)
        self.N = N

    def S2V(self, S):
        return set.union(*[self.N[i] for i in S])


class VectorData(Data):
    def __init__(self, V: np.ndarray, target: np.array, sim=None, nsample=100):
        '''
        target: target vector
        sim: calculate similarity b/w two vectors, cosine by default
        '''
        super().__init__()
        self.n = len(V)
        self._V = V
        self.V = np.arange(self.n)
        self.tar = target
        self.sim = lambda a,b: np.dot(a, b) / (norm(a)*norm(b)) - 0.5 \
            if sim is None else sim

        # A random sample of V for evaluation of f
        if nsample >= self.n:
            self.samples = self._V
        else:
            idxs = np.random.choice(self.V, size=nsample, replace=False)
            self.samples = self._V[idxs]
        # Pre-compute sims
        self.sims = []
        for v in self._V:
            sims_v = [self.sim(u,v) for u in self.samples]
            self.sims.append(sims_v)
        self.sims_tar = [self.sim(v,self.tar) for v in self._V]

    def f_rel(self, S):
        return sum([self.sims_tar[i] for i in S])

    def f_div(self, S):
        if len(S) == 0:
            return 0
        val = 0
        maxs = [-1] * len(self.samples)
        for i in S:
            sims_i = self.sims[i]
            for j in range(len(maxs)):
                maxs[j] = max(maxs[j], sims_i[j])
        return np.mean(maxs), maxs

    def next_demand(self, tradeoff: float, k: int, weight: float=1):
        '''
        f(S) = (1-t) relevant_S + t * diversity_S
        k: cardinality
        '''
        if tradeoff is None:
            return

        def _f(S, i, t, kmax, w, extra=False):
            if len(S) == 0:
                if extra:
                    return 0, (None,None)
                return 0
            S = set(S)
            R, _, renew = self.cache(S, i)
            if R is not None: # cache hit
                f1, maxs = _
                if len(R) == 0:
                    f2 = np.mean(maxs)
                else:
                    _, maxsR = self.f_div(R)
                    maxs = [max(a,b) for a,b in zip(maxs, maxsR)]
                    f1 = self.f_rel(R) + f1
                    f2 = np.mean(maxs)
            else:
                f1 = self.f_rel(S)
                f2, maxs = self.f_div(S)

            if renew:
                self.caches[i] = (S,(f1,maxs)) # renew cache

            val = (1-t) * f1 + t * f2*kmax
            val = val * w
            if extra:
                return val, (f1/len(S), f2)
            return val

        self.caches.append((set(),
                            (0, [-1]*len(self.samples))
                            )) # (last S, f(S))
        self.F.append((partial(_f, i=len(self.F), t=tradeoff, kmax=k, w=weight), k))

