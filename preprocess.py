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

def preprocess_box(data_dir='data/wide_trails'):
    """Mouse coordinates are: 
        snout -> r ear -> l ear -> shoulder -> tail base
        -> ll box -> lr box -> ul box -> ur box
    """

    data_dir = Path(data_dir)
    mouse_regex = 'Mus_w\d+.npy'
    num_regex = '\d+'

    data = defaultdict(dict)
    for path in data_dir.iterdir():
        if re.search(mouse_regex, str(path)):
            num_match = re.search(num_regex, str(path))
            mouse_id = num_match[0]
            raw_arr = np.load(path)

            raw_arr[11,:][raw_arr[11,:] > 40] = 40   # clamp nonsense values in trail y coordinates

            # TODO: test, and filter NaN's
            raw_trail = drop_nan_col(raw_arr[10:12,:])
            raw_mouse = drop_nan_col(np.concatenate((raw_arr[:10,:], raw_arr[12:,:]), axis=0))

            data[mouse_id]['trail'] = raw_trail
            data[mouse_id]['mouse'] = raw_mouse
        else:
            print(f'warn: skipping {path}')
            continue

    data = pd.DataFrame(data).T
    pd.to_pickle(data, 'data/df.pkl')
    return data


def drop_nan_col(x):
    return x[:, ~np.any(np.isnan(x), axis=0)]


def preprocess_sep(data_dir=Path('data/')):
    """Mouse coordinates are: 
        snout -> r ear -> l ear -> shoulder -> tail base
    """

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


def segment(mouse, trail, srate=3, window=5, flatten=False):
    skip = sampling_rate // srate
    mouse = mouse[:,::skip]

    mouse, box = mouse[:10], mouse[10:]
    # r_dists, l_dists, u_dists, d_didsts, t_dists = _get_perp_dists(mouse, box, trail)
    all_dists = _get_perp_dists(mouse, box, trail)

    exs = []
    for i in range(mouse.shape[1] - window):
        start_idx = i
        stop_idx = i + window

        m = mouse[:, stop_idx-1]
        trans = _make_trans(m)

        xs = trans(mouse[:, start_idx:stop_idx])
        y = trans(mouse[:2, [stop_idx]])

        trans_dists = [trans(dist[:, start_idx:stop_idx]) for dist in all_dists]

        # xs = np.concatenate((xs, perp_dists[:,start_idx:stop_idx], ts, box[:,start_idx:stop_idx]))
        xs = np.concatenate((xs, *trans_dists))  # TODO: test <-- STOPPED HERE
        exs.append((xs, y))

    X, y = zip(*exs)
    X, y = np.array(X), np.array(y)

    if flatten == True:
        X = X.reshape(X.shape[0], X.shape[1] * window)
        y = np.squeeze(y)
    
    return X, y


def _get_perp_dists(mouse, box, trail):
    snout = mouse[:2,:]
    box_ll = box[:2,:]
    box_ur = box[-2:,:]

    r_dists = np.stack((box_ll[0], snout[1]))
    l_dists = np.stack((box_ur[0], snout[1]))
    u_dists = np.stack((snout[0], box_ur[1]))
    d_dists = np.stack((snout[0], box_ll[1]))

    # r_dists, u_dists = np.abs(box_ur - snout)
    # l_dists, d_dists = np.abs(box_ll - snout)

    m_norm = np.linalg.norm(snout, axis=0, keepdims=True)
    t_norm = np.linalg.norm(trail, axis=0, keepdims=True)
    t_dists = np.sqrt(m_norm.T**2 - 2 * snout.T @ trail + t_norm**2)

    t_idx = np.argmin(t_dists, axis=1)
    t_dists = trail[:,t_idx] 
    # most_dists = np.stack((r_dists, u_dists, l_dists, d_dists))
    return r_dists, u_dists, l_dists, d_dists, t_dists   # NOTE: origin will be subtracted above


def _make_trans(m):
    origin = m[:2]
    tail = m[-2:]
    dir_ = tail - origin
    if dir_[0] == dir_[1] == 0:
        theta = 0
    else:
        theta = np.arctan(dir_[0] / dir_[1])

        if dir_[1] > 0:
            theta += np.pi

    def trans(x):
        rot_mat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        # rot_mat = np.eye(2)

        n_reps = x.shape[0] // 2
        rot_mat = block_diag(*(n_reps * [rot_mat]))
        origin = np.tile(m[:2], n_reps)
        return rot_mat @ (x - origin.reshape(-1, 1))
    
    # return lambda x: x
    return trans

if __name__ == '__main__':
    data = pd.read_pickle('data/df.pkl')
    X, y = segment(data.mouse[0], data.trail[0])
    idx = 113
    # TODO: check mouse will sometimes exceed bounds of box

    mx = X[idx, 0, :]
    my = X[idx, 1, :]

    px = mx[-1]
    py = my[-1]

    rx, ry, ux, uy, lx, ly, dwx, dwy, tx_dist, ty_dist = X[idx, 10:, -1]
    # llx, lly, lrx, lry, ulx, uly, urx, ury = X[idx, 16:,-1]
    # print(llx)

    dx = X[idx,0,-1] - X[idx,8,-1]
    dy = X[idx,1,-1] - X[idx,9,-1]

    plt.plot(mx, my)

    plt.arrow(px, py, y[idx, 0, 0]-px, y[idx, 1, 0]-py, head_width=0.5, head_length=0.5, color='red')
    # plt.arrow(px, py, r_dist, 0, color='gray')
    # plt.arrow(px, py, -l_dist, 0, color='gray')
    # plt.arrow(px, py, 0, u_dist, color='gray')
    # plt.arrow(px, py, 0, -d_dist, color='gray')

    plt.arrow(px, py, rx, ry, color='gray', alpha=0.5)
    plt.arrow(px, py, ux, uy, color='gray', alpha=0.5)
    plt.arrow(px, py, lx, ly, color='gray', alpha=0.5)
    plt.arrow(px, py, dwx, dwy, color='gray', alpha=0.5)

    plt.arrow(px, py, tx_dist, ty_dist, color='orange')

    # plt.scatter([llx, lrx, ulx, urx], [lly, lry, uly, ury])

    # circ = plt.Circle((px, py), t_dist, fill=False, clip_on=True)
    # plt.gca().add_patch(circ)


    # plt.scatter(data.trail[0][0], data.trail[0][1], s=0.1)
    # plt.xlim((llx, lrx))
    # plt.ylim((lly, uly))
    # plt.arrow(px, py, dx, dy, color='magenta')

    # plt.arrow(0, 0, 0, 1)
    # plt.xlim((-20, 20))
    # plt.ylim((-20, 20))



# <codecell>
    X, y = segment(data.mouse[0], data.trail[0], flatten=True)
    X_, y_ = segment(data.mouse[1], data.trail[1], flatten=True)

    X = np.concatenate((X, X_), axis=0)
    y = np.concatenate((y, y_), axis=0)

    X = X.reshape(X.shape[0], -1)
    y = y.squeeze()

    model = Lasso(alpha=0.1)
    model.fit(X, y)

    print('TRAIN', model.score(X, y))

    X_test, y_test = segment(data.mouse[2], data.trail[2], flatten=True)
    X_test = X_test.reshape(X_test.shape[0], -1)
    y_test = y_test.squeeze()

    print('TEST', model.score(X_test, y_test))
# %%
