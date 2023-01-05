import pytest
import numpy as np
from functools import partial

from msr.base import Seq
from msr.data import CoverageData, VectorData, DemandStream, ItemStream
from msr.msr import MSRF, MSRI


@pytest.fixture
def vec():
    np.random.seed(42)
    V = np.random.rand(5, 2)
    ks = [1,2]
    tds = [0.1, 1.0]
    return (V,tds,ks)


@pytest.fixture
def cov():
    V = np.arange(5)
    subsets = [
        [0,1,2],
        [2,3,4],
    ]
    ks = [1,2]
    return (V,subsets,ks)


def input_msrf(data, DataModel):
    np.random.seed(42)
    V,subsets,ks = data
    if DataModel is CoverageData:
        dat = DataModel()
    elif DataModel is VectorData:
        dat = DataModel(V, V[0])
    for s,k in zip(subsets,ks):
        dat.next_demand(s,k)
    dat.next_demand(None,None)
    return dat, DemandStream(dat.F) # [],[],[f1],[f2],[],...


def input_msri(data, DataModel):
    np.random.seed(42)
    V,subsets,ks = data
    if DataModel is CoverageData:
        dat = DataModel()
    elif DataModel is VectorData:
        dat = DataModel(V, V[0])
    for s,k in zip(subsets,ks):
        dat.next_demand(s,k)
    dat.next_demand(None,None)
    return dat, ItemStream(dat.V) # [1, 4, 2, 0, 3]


def test_seq():
    seq = Seq()
    for i in range(0,10,2):
        seq.add(i,i*2)
    assert seq.at(0) == 0
    assert seq.at(2,2) == [2*2]
    assert seq.at(1,5) == [2*2,4*2]
    assert seq.at(0,4) == [0,2*2,4*2]


def test_coverage_data(cov):
    V,subsets,ks = cov
    dat = CoverageData()
    for s,k in zip(subsets,ks):
        dat.next_demand(s,k)
    dat.next_demand(None,None)

    assert np.isclose(sum(dat.V - V), 0)
    (f1,_), (f2,_) = dat.F
    assert f1(set()) == 0
    assert np.isclose(f1(set([1,2])), 2/3)
    assert np.isclose(f2(set([1,2])), 1/3)
    assert np.isclose(f2(set(V)), 1)

    np.random.seed(42)
    stream = DemandStream(dat.F)
    F = []
    for t, Ft in enumerate(stream):
        F = F + Ft
    assert len(F) == len(dat.F)


def test_msrf(cov):
    dat, stream = input_msrf(cov, CoverageData)
    np.random.seed(42)
    seq = Seq()
    method = 'greedy'
    model = MSRF(dat.V, method)
    for t, Ft in enumerate(stream):
        v = model.next(t, Ft)
        if t == 0:
            assert len(model.At) == 0
        if t == 2:
            assert len(model.At) == 1
        if t == 3:
            assert len(model.At) == 1
        if t == 4:
            assert len(model.At) == 1
        if t == 5:
            assert len(model.At) == 0
        seq.add(t, v)
    #print(seq.items) # [0, 0, 0, 2, 3, 0, ...
    assert np.isclose(model.obj(seq), 1/3+2/3)


def test_msri(cov):
    dat, stream = input_msri(cov, CoverageData)
    np.random.seed(42)
    method = 'exc'
    model = MSRI(dat.F, len(dat.V), method)
    for t, v in enumerate(stream):
        v = model.next(t, v)
    seq = model.next(None,None)
    assert seq.seq == [2,4]
    assert np.isclose(1, 1/3+2/3)


def test_msri_greedyo(cov):
    dat, stream = input_msri(cov, CoverageData)
    np.random.seed(42)
    method = 'greedyo'
    model = MSRI(dat.F, len(dat.V), method)
    for t, v in enumerate(stream):
        v = model.next(t, v)
    seq = model.next(None,None)
    assert seq.at(0) == 2
    assert np.isclose(1, 2/3+1/3)


def test_msri_vec(vec):
    dat, stream = input_msri(vec, VectorData)
    np.random.seed(42)
    method = 'exc'
    model = MSRI(dat.F, len(dat.V), method)
    for t, v in enumerate(stream):
        v = model.next(t, v)
    seq = model.next(None,None)
    assert seq.seq == [1,4]
    #assert np.isclose(model.obj(seq), 5.986)
    assert np.isclose(model.obj(seq), 1.275013731418954) # cosine munis 0.5


def test_cache():
    n = 10000
    V = np.arange(n)

    dat = CoverageData()
    dat.next_demand(V, n)
    dat.next_demand(None,None)

    i = 0
    f,_ = dat.F[i]
    s = set(np.arange(1000))
    R,D,renew = dat.cache(s, i)
    assert R == s
    assert renew
    assert np.isclose(f(s), 1000/n)
    R,D,renew = dat.cache(s, i)
    assert R == set()
    assert not renew

    s = set(np.arange(1001))
    R,D,renew = dat.cache(s, i)
    assert R == {1000}
    assert not renew
    assert np.isclose(f(s), 1001/n)
    R,D,renew = dat.cache(s, i)
    assert R == {1000}
    assert not renew

    s = set(np.arange(1002))
    R,D,renew = dat.cache(s, i)
    assert R == {1000,1001}
    assert renew
    assert np.isclose(f(s), 1002/n)
    R,D,renew = dat.cache(s, i)
    assert R == set()
    assert not renew

