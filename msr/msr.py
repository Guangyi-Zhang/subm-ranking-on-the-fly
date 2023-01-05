import numpy as np

from msr.base import Seq, MSR


class MSRI(MSR):
    def __init__(self, F, n, method, V=None, exc_ratio=2):
        '''
        F: list of (f,k)
        V for omniscient method.
        TODO: p-matroid
        '''
        super().__init__()
        self.F = F
        self.kmax = max([k for f,k in F])
        self.n = n
        self.method = method
        self.V = V
        self.seq = Seq()
        self.weight = dict() # per (r,v), for exc
        self.excr = exc_ratio

    def next(self, t: int, objects):
        v = objects
        if v is None:
            return self.output()

        if self.method == 'exc':
            for r in range(self.kmax): # assume edges of v arrive in order of r
                #val = self.val_rv(r, v) # seems ok with feasible T or F
                val = self.val_rv(r, v, feasible=True) # seems ok with feasible T or F

                flag = self.seq.exists(r)
                if flag:
                    vold = self.seq.at(r)
                    w = self.weight[(r, self.seq.at(r))]
                    if val < self.excr * w:
                        continue

                self.seq.add(r, v)
                self.weight[(r, v)] = val

                if flag:
                    v = vold # avoid dropping v_old
                else:
                    break

        elif self.method == 'random':
            pass
        elif self.method == 'top':
            pass
        elif self.method == 'greedyo':
            pass
        else:
            raise Exception

    def val_rv(self, r, v, feasible=False):
        '''
        feasible=True: ignore the current item at r if occupied
        '''
        val = 0
        for i,(f,k) in enumerate(self.F):
            if r < k:
                items1 = self.seq.at(0, r-1 if feasible else r)
                items2 = self.seq.at(r+1, k-1)
                items = set(items1 + items2)
                val = val + f(items.union({v})) - f(items)
        return val

    def greedyo(self):
        seq = Seq()
        items = set()
        vals = [1e7] * self.n
        for r in range(self.kmax):
            print(r)
            gmax = -1
            for v in range(self.n):
                if seq.vexists(v):
                    continue
                if vals[v] <= gmax: # lazy eval
                    continue
                g = sum([f(items.union({v})) - f(items) for f, k in self.F if k > r])
                vals[v] = g
                if gmax < g:
                    gmax = g
            vmax = np.argmax(vals)
            vals[vmax] = -1
            seq.add(r, vmax)
            items.add(vmax)

        return seq

    def output(self):
        if self.method == 'greedyo':
            self.seq = self.greedyo()
            return self.seq
        elif self.method == 'top':
            vals = []
            for v in range(self.n):
                g = sum([f({v}) for f, k in self.F])
                vals.append(g)
            for r, i in enumerate(reversed(np.argsort(vals))):
                if r > self.kmax: break
                self.seq.add(r, i)
            return self.seq
        elif self.method == 'random':
            items = np.random.choice(self.n, size=self.kmax, replace=False)
            for r,v in enumerate(items):
                self.seq.add(r, v)
            return self.seq
        elif self.method == 'exc':
            return self.seq

        raise

    def obj(self, seq: Seq, extra=False):
        val = 0
        extras = []
        for f,k in self.F:
            items = seq.at(0, k-1)
            if extra:
                fval,_ = f(set(items), extra=extra)
                val = val + fval
                extras.append(_)
            else:
                val = val + f(set(items))

        if extra:
            return val, extras
        return val


class MSRF(MSR):
    def __init__(self, V, method, F=None, topk: int=10):
        super().__init__()
        self.V = V
        self.method = method
        self.At = set()
        self.id2F = dict()
        self.seq = Seq()
        self.vals = list() # list of dict, one for each v, funcid2val
        for _ in self.V:
            self.vals.append(dict())

        # user F, omniscient
        self.topk = list()
        self.loop = list() # topk in use
        if self.method.startswith('top'):
            k = int(self.method[3:]) if len(self.method) > 3 else topk
            vals = []
            for v in self.V:
                g = sum([f({v}) for f, k in F])
                vals.append(g)
            for i in reversed(np.argsort(vals)):
                if len(self.topk) >= k: break
                self.topk.append(self.V[i])

    def active(self, t: int, newids: list, At: set):
        # Update active functions in At
        torm = []
        for id_ in At:
            f,k,a = self.id2F[id_]
            if k <= t-a:
                torm.append(id_)
        for id_ in torm:
            At.remove(id_)
        for id_ in newids:
            At.add(id_)

    def next(self, t: int, objects):
        # Mark each f, only once
        F = objects
        ids = []
        for i,(f,k) in enumerate(F):
            id_ = (t,i)
            self.id2F[id_] = (f,k,t)
            ids.append(id_)

        # Update active F
        self.active(t, ids, self.At)

        # Decide t-th item
        v = self.pick(t, self.At)
        self.seq.add(t, v)
        return v

    def pick(self, t: int, At: set):
        if self.method == 'random':
            i = np.random.randint(0,len(self.V))
            return self.V[i]
        elif self.method.startswith('top'):
            if len(self.loop) == 0:
                self.loop = [v for v in self.topk]
            return self.loop.pop()
        elif self.method == 'greedy':
            #vals = [self.val_t(t,v,self.vals[j],At,self.seq) for j,v in enumerate(self.V)]
            vmax, jmax = -1, None
            for j,v in enumerate(self.V):
                val = 0
                vd = self.vals[j]

                oldids = set()
                for id_ in vd.keys():
                    if id_ not in At:
                        oldids.add(id_)
                for id_ in oldids:
                    vd.pop(id_, None)

                newids = set()
                for id_ in At:
                    if id_ not in vd:
                        newids.add(id_)
                        vd[id_] = self.val_tvf(t,v,id_,self.seq)
                    val = val + vd[id_]

                if val > vmax:
                    val = 0 # compute true val
                    for id_ in At:
                        if id_ not in newids:
                            vd[id_] = self.val_tvf(t,v,id_,self.seq)
                        val = val + vd[id_]
                    vmax = val
                    jmax = j

            return self.V[jmax]

        raise Exception

    def val_tvf(self, t: int, v: int, fid: int, seq: Seq):
        f,k,a = self.id2F[fid]
        items = seq.at(a, t-1)
        g = f(set(items + [v])) - f(set(items))
        return g

    def obj(self, seq: Seq):
        val = 0
        for f,k,a in self.id2F.values():
            items = seq.at(a, a+k-1)
            val = val + f(set(items))
        return val
