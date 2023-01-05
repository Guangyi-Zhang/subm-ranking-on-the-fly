import numpy as np
import pandas as pd
import pickle
from collections import defaultdict


rng = np.random.default_rng(12345)
fdata = '/NOBACKUP/guaz/msr-datasets/train_triplets.txt'


def sample_users(nsamp, nsz, like_thr=1):
    users = set()
    with open(fdata, 'r') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            user, song, count = line.split()
            users.add(user)

    users = list(users)
    print('#user: ', len(users))

    samp = []
    for i in range(nsamp):
        idx = rng.choice(range(len(users)), size=nsz, replace=False)
        usamp = [users[j] for j in idx]
        samp.append(dict([(u, []) for u in usamp]))


    for i in range(nsamp):
        songs = set()
        s = samp[i]
        with open(fdata, 'r') as fin:
            while True:
                line = fin.readline()
                if not line:
                    break
                user, song, count = line.split()
                count = int(count)
                if user in s and count > like_thr:
                    s[user].append(song)
                    songs.add(song)

        pickle.dump([len(songs), s], open(f'datasets/music.pkl', 'wb'))


def test_song_overlap():
    with open('datasets/music.pkl', 'rb') as fin:
        n, s = pickle.load(fin)
        m = len(s)
        print(f'#songs={n}, #users={m}')

        counter = defaultdict(int)
        for u,sgs in s.items():
            for song in sgs:
                counter[song] += 1

        hist = defaultdict(int)
        for sg, c in counter.items():
            hist[c] += 1
        print(hist)


if __name__ == '__main__':
    #sample_users(nsamp=1, nsz=10000)
    test_song_overlap()
