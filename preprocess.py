"""
Preprocess raw data files from Siddharth
"""

# <codecell>
from collections import defaultdict
from pathlib import Path
import re

import jax
from jax import random, numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax


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
pd.to_pickle(data, 'data/df.pkl')

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
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(2)(x)
        return x

key, k1, k2 = random.split(random.PRNGKey(42), 3)

model = MLP()
params = model.init(k1, jnp.empty((1, 1200)))

X, y = segment(data.mouse[0], data.trail[0])
X_, y_ = segment(data.mouse[1], data.trail[1])

X = np.concatenate((X, X_), axis=0)
y = np.concatenate((y, y_), axis=0)

X_test, y_test = segment(data.mouse[2], data.trail[2])

out = model.apply(params, X)
jnp.mean((out - y) ** 2)

# %%
@jax.jit
def compute_mse(state, batch):
    X, y = batch
    preds = state.apply_fn(state.params, X)
    loss = jnp.mean((preds - y) ** 2)
    return loss

@jax.jit
def compute_r2(state, batch):
    mse = compute_mse(state, batch)
    y_var = jnp.mean((y - jnp.mean(y)) ** 2)
    return 1 - mse / y_var

def create_train_state(model, rng, lr=1e-4):
    params = model.init(rng, jnp.empty((1, 300 * 4)))

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(lr)
    )

@jax.jit
def train_step(state, batch):
    X, y = batch

    def loss_fn(params):
        preds = state.apply_fn(params, X)
        loss = jnp.mean((preds - y) ** 2)
        return loss
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state

state = create_train_state(model, k2)

for i in range(10000):
    state = train_step(state, (X, y))

    if i % 100 == 0:
        train_r2 = compute_r2(state, (X, y))
        test_r2 = compute_r2(state, (X_test, y_test))
        print(f'Iter {i}: train_r2 = {train_r2:.4f}   test_r2 = {test_r2:.4f}')

# %%
