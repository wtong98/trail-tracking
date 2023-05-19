"""
Preprocess raw data files from Siddharth
"""

# <codecell>
from collections import defaultdict
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from sklearn.linear_model import *

sampling_rate = 30

def preprocess(data_dir=Path('data/')):
    """Mouse coordinates go snout -> r ear -> l ear -> shoulder -> tail base"""
    mouse_regex = 'Test_mus_\d+.*'
    trail_regex = 'Trail_\d+.*'
    num_regex = '\d+'

    data = defaultdict(dict)

    for path in data_dir.iterdir():
        idx = 'other'
        if re.search(mouse_regex, str(path)):
            idx = 'mouse'
        elif re.search(trail_regex, str(path)):
            idx = 'trail'
        else:
            print(f'warn: skipping {path}')
            continue

        num_match = re.search(num_regex, str(path))
        mouse_id = num_match[0]
        raw_arr = np.load(path)

        if idx == 'trail':
            raw_arr[1,:][raw_arr[1,:] > 40] = 0   # clamp nonsense values

        data[mouse_id][idx] = raw_arr[:,3:]  # brute-force nan drop

    data = pd.DataFrame(data).T
    pd.to_pickle(data, 'data/df.pkl')


def segment(mouse, trail, track_delay=1, srate=3, window=5):
    skip = sampling_rate // srate
    delay = int(track_delay * sampling_rate)
    print('DELAY', delay)  # TODO: switch to coordinates closest to the mouse snout <-- STOPPED HERE

    max_size = min(trail.shape[1], mouse.shape[1])  #TODO: wrap in bounding box
    
    mouse = mouse[:,:(max_size-delay):skip]
    trail = trail[:,delay:max_size:skip]
    print(trail.shape)
    print(trail[:,200:205])

    exs = []
    for i in range(mouse.shape[1] - window):
        start_idx = i
        stop_idx = i + window

        m = mouse[:, stop_idx-1]
        trans = _make_trans(m)

        xs = trans(mouse[:, start_idx:stop_idx])
        ts = trans(trail[:, start_idx:stop_idx])
        y = trans(mouse[:2, [stop_idx]])

        xs = np.concatenate((xs, ts), axis=0)
        exs.append((xs, y))

    X, y = zip(*exs)
    
    return np.array(X), np.array(y)


def _make_trans(m):
    origin = m[:2]
    tail = m[-2:]
    dir_ = tail - origin
    if dir_[0] == dir_[1] == 0:
        theta = 0
    else:
        theta = np.arctan(dir_[0] / dir_[1])

        if dir_[1] > 0:
            # theta = -theta - np.sign(dir_[0]) * (np.pi / 2)
            theta = np.sign(dir_[0]) * (theta - np.pi)

    def trans(x):
        rot_mat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        n_reps = x.shape[0] // 2
        rot_mat = block_diag(*(n_reps * [rot_mat]))
        origin = np.tile(m[:2], n_reps)
        return rot_mat @ (x - origin.reshape(-1, 1))
    
    # return lambda x: x
    return trans

# TODO: align regressors more carefully
def segment_old(mouse, trail, window=300, stack=True):
    snout = mouse[:2,:]
    snout = snout[:,:trail.shape[1]]

    exs = []
    for i in range(snout.shape[1] - window):
        start_idx = i
        stop_idx = i + window
        xs = snout[:, start_idx:stop_idx]
        ts = trail[:, start_idx:stop_idx]
        y = snout[:, stop_idx].reshape(-1, 1)

        curr_pos = xs[:,-1].reshape(-1, 1)

        xs = np.concatenate((xs, ts), axis=1)
        exs.append((xs - curr_pos, y - curr_pos))

    X, y = zip(*exs)

    if stack:
        X = np.stack(X).reshape(len(exs), -1)
        y = np.stack(y).reshape(len(exs), -1)
    
    return X, y


if __name__ == '__main__':
    data = pd.read_pickle('data/df.pkl')
    X, y = segment(data.mouse[0], data.trail[0])
    idx = 290

    mx = X[idx, 0, :]
    my = X[idx, 1, :]

    dx = X[idx,0,-1] - X[idx,-4,-1]
    dy = X[idx,1,-1] - X[idx,-3,-1]

    tx = X[idx, -2, :]
    ty = X[idx, -1, :]

    plt.plot(mx, my)
    plt.plot(tx, ty)

    plt.arrow(0, 0, y[idx, 0, 0], y[idx, 1, 0], head_width=0.5, head_length=0.5, color='red')
    plt.arrow(0, 0, dx, dy, color='magenta')

    plt.arrow(0, 0, 0, 1)
    # plt.xlim((-20, 20))
    # plt.ylim((-20, 20))



    # <codecell>
    # X, y = segment(data.mouse[0], data.trail[0])
    # X_, y_ = segment(data.mouse[1], data.trail[1])

    # X = np.concatenate((X, X_), axis=0)
    # y = np.concatenate((y, y_), axis=0)

    # model = Lasso(alpha=0.1)
    # model.fit(X, y)

    # model.score(X, y)

    # X_test, y_test = segment(data.mouse[2], data.trail[2])
    # model.score(X_test, y_test)