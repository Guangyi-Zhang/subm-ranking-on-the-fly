# online-msr-code

## Repo structure

* `msr/`: algorithms and baselines
* `data_scripts/`: scripts that process datasets
* `tests/`: test cases

Datasets can be download from

* http://millionsongdataset.com/
* http://snap.stanford.edu/data/github-social.html
* http://www.sogou.com/labs/resource/q.php
* https://nlp.stanford.edu/projects/glove/

## Examples

```python
import numpy as np

from msr.base import Seq
from msr.data import CoverageData, NetworkData, VectorData, DemandStream, ItemStream
from msr.msr import MSRF, MSRI


n, m, ss = 100, 10, 20
streaming_type = 'MSRF'
method = 'greedy' # for MSRF, use random,greedy,topK (for any integer K)
#streaming_type = 'MSRI'
#method = 'exc' # for MSRI, use random,top,exc,greedyo

np.random.seed(42) # fix random state

# Prepare synthetic data
V = np.arange(n)
subsets = [np.random.choice(n, size=ss, replace=False)
           for _ in range(m)]
ks = np.random.randint(1, ss, size=m)
dat = CoverageData()
for s, k in zip(subsets, ks):
    if len(s) == 0:
        continue
    dat.next_demand(s, k)
dat.next_demand(None, None)

# Load models
if streaming_type == 'MSRF':
    model = MSRF(dat.V, method, F=dat.F)
    stream = DemandStream(dat.F)
    seq = Seq()
    for t, Ft in enumerate(stream):
        v = model.next(t, Ft)
        seq.add(t, v)
elif streaming_type == 'MSRI':
    model = MSRI(dat.F, len(dat.V), method=method, exc_ratio=2)
    stream = ItemStream(dat.V)
    for t, v in enumerate(stream):
        model.next(t, v)
    seq = model.next(None, None)

# Print output
obj = model.obj(seq)
print(f'model {streaming_type}, method {method}, objective: {obj}')
print(f'sequence: {seq.seq}')
```

## Tests

```
py.test
```


# License
This project is licensed under the terms of the MIT license.
