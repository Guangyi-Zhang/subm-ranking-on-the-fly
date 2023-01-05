import abc
from typing import Callable
import numpy as np
from bisect import bisect
from collections import defaultdict


class Seq(abc.ABC):
    '''
    Support range search: dict is enough as k is queried ranges are typically small.
    '''
    def __init__(self):
        self.r2v = dict()
        self.items = set()

    def exists(self, r: int):
        return r in self.r2v

    def vexists(self, v: int):
        return v in self.items

    def at(self, beg: int, end: int=None):
        '''[beg,end] '''
        if end is None:
            return self.r2v[beg]

        items = []
        for r in range(beg, end+1):
            if r in self.r2v:
                items.append(self.r2v[r])
        return items

    def add(self, r: int, v):
        ''' Replace if r exists'''
        self.r2v[r] = v
        self.items.add(v)

    @property
    def seq(self):
        ''' Squeeze empty ranks '''
        rs = sorted(self.r2v.keys())
        return [self.r2v[r] for r in rs]


class MSR(abc.ABC):
    '''
    MSR-I: Exc, omniscent greedy, rank-1 utility, random
    MSR-F: greedy, top-k, random, omniscent greedy
    '''
    def __init__(self):
        pass

    def next(self, t: int, objects):
        pass

    def obj(self, seq: Seq):
        pass


class Data(abc.ABC):
    '''
    API: f(v|S)=f(v,S)
    Music: (#songs, users=[[liked songs 1 2 3],])
    Network:
    Web pages:
    Recommended vectors: f(v|S) = u'v + lambda * sum_w max_{z in S+v} w'z
    '''
    def __init__(self):
        self.F = list()
        self.caches = list() # list of (last S, data wrt S)

    def cache(self, S, i):
        renew = False
        S1, D1 = self.caches[i]
        if len(S.intersection(S1)) == len(S1):  # cache hit
            R = S.difference(S1)
            if len(R) > 1:  # false when f(v|S)
                renew = True
            return R, D1, renew
        else:
            return None, None, False

    def makeF(self):
        pass


class Stream(abc.ABC):
    def __init__(self):
        pass

    def __iter__(self):
        pass
