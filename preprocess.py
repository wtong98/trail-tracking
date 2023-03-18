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
from sklearn.linear_model import *

# <codecell>
data_dir = Path('data/')

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

# <codecell>
# TODO: align regressors more carefully
def segment(mouse, trail, window=300, stack=True):
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

X, y = segment(data.mouse[0], data.trail[0])
idx = 5150

mx, tx, my, ty = np.split(X[idx], 4)
plt.plot(mx, my)
plt.plot(tx, ty)
# <codecell>
X, y = segment(data.mouse[0], data.trail[0])
X_, y_ = segment(data.mouse[1], data.trail[1])

X = np.concatenate((X, X_), axis=0)
y = np.concatenate((y, y_), axis=0)

model = Lasso(alpha=0.1)
model.fit(X, y)

# model.score(X, y)

X_test, y_test = segment(data.mouse[2], data.trail[2])
model.score(X_test, y_test)

# <codecell>

t = data.trail[1]
m = data.mouse[1]

t[1,:][t[1,:] > 35] = 0
# plt.plot(t[0,:])
plt.plot(m[1,:])
plt.plot(t[1,:])

plt.xlim((1500, 2000))


# <codecell>

start = 6000
length = 509
time_idx = np.arange(start, start + length)
mouse_data = np.load('data/Test_mus_1.np.npy')
mouse_data = mouse_data[:, 3:][:,time_idx]

trail_data = np.load('data/Trail_1.np.npy')
trail_data[:, 3:][:,time_idx]

plt.plot(trail_data[0], trail_data[1])
plt.plot(mouse_data[0], mouse_data[1])

plt.ylim((0, 35))
plt.xlim((400, 600))
# %%
