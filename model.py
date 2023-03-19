"""
Models for explaining trail tracking data

author: William Tong (wtong@g.harvard.edu)
date: March 6, 2023
"""

# <codecell>
import numpy as np
from torch.utils.data import Dataset

import jax
from jax import random, numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

class MlpModel:
    def __init__(self, rng_seed=1130) -> None:
        self.key, k = random.split(random.PRNGKey(rng_seed), 2)
        self.model = MLP()
        self.state = create_train_state(self.model, k)

    def fit(self, X, y, X_test=None, y_test=None, train_iters=10_000):
        batch = (X, y)

        for i in range(train_iters):
            self.state = train_step(self.state, batch)

            if i % 100 == 0:
                train_r2 = compute_r2(self.state, (X, y))
                if X_test and y_test:
                    test_r2 = compute_r2(self.state, (X_test, y_test))
                else:
                    test_r2 = 0

                print(f'Iter {i}: train_r2 = {train_r2:.4f}   test_r2 = {test_r2:.4f}')

    def predict(self, X):
        return self.model.apply(self.state.params, X)


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

