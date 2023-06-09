"""
Experiments with 1D analysis
"""

# <codecell>
from dataclasses import dataclass, field
import typing

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso

from model import MlpModel
from preprocess import segment

data = pd.read_pickle('data/df.pkl')

X, y = segment(data.mouse[0], data.trail[0])
X_, y_ = segment(data.mouse[1], data.trail[1])

X = np.concatenate((X, X_), axis=0)
y = np.concatenate((y, y_), axis=0)

X_test, y_test = segment(data.mouse[2], data.trail[2])

# <codecell>
@dataclass
class Case:
    name: str
    model: typing.Any
    train_r2: float = -999
    test_r2: float = -999
    args: dict = field(default_factory=dict)

cases = [
    Case(name='Linear', model=LinearRegression()),
    Case(name='Lasso', model=Lasso(alpha=0.1)),
    Case(name='MLP', model=MlpModel(), args={'X_test': X_test, 'y_test': y_test}),
]

for case in cases:
    print(f'Processing: {case.name}')
    case.model.fit(X, y, **case.args)
    case.train_r2 = case.model.score(X, y)
    case.test_r2 = case.model.score(X_test, y_test)


# <codecell>
plot_df = pd.DataFrame(cases)
plot_df = plot_df.melt(id_vars=['name'], value_vars=['train_r2', 'test_r2'], value_name='r2')
plot_df['r2'] = plot_df['r2'].astype(float)

g = sns.barplot(plot_df, x='name', y='r2', hue='variable')
g.legend().set_title('')
g.set_xlabel('')
g.set_ylabel(r'$R^2$')
g.set_title(r'$R^2$ across 3 models on 1D data')
plt.savefig('fig/1d_comparison.png')

# <codecell>

# TODO: cleanup and animate with bounding boxes
def animate(model, X_test, mouse, trail, n_preds=250, start_idx=0, save_path='anim.mp4'):
    mouse = np.copy(mouse)
    trail = np.copy(trail)

    xs = X_test[start_idx,:]
    all_xs = [xs]

    origin = mouse.T[start_idx+300, :2]

    for i in range(n_preds):
        ys = model.predict(xs.reshape(1, -1)).flatten()
        mx, tx, my, ty = np.split(np.copy(xs), 4)
        origin += ys

        mxn, txn, myn, tyn = np.split(X_test[start_idx+i+1],4)

        mx[:-1] = mx[1:]
        mx[-1] = ys[0]
        mx = mx - ys[0]

        my[:-1] = my[1:]
        my[-1] = ys[1]
        my = my - ys[1]

        trail_seg = trail.T[start_idx+i:start_idx+i+300]
        tx = trail_seg[:,0] - origin[0]
        ty = trail_seg[:,1] - origin[1]
        # print('ORIGIN', origin)
        # print('TRAIL', trail_seg[-1,1])
        # print('TY', ty[-1])

        xs = np.concatenate((mx, tx, my, ty))
        all_xs.append(xs)

    frames = list(zip(all_xs, X_test[start_idx:start_idx+n_preds]))

    def plot_frame(frame):
        plt.clf()

        xs, X_test = frame
        mx, tx, my, ty = np.split(xs, 4)
        plt.plot(mx, my)

        mx_t, _, my_t, _ = np.split(X_test, 4)

        x_adj = mx_t - mx
        y_adj = my_t - my

        plt.plot(mx_t - x_adj[0], my_t - y_adj[0], alpha=0.7)
        plt.plot(tx, ty, color='red', alpha=0.5)

        plt.xlim((-22, 80))
        plt.ylim((-15, 15))


    ani = FuncAnimation(plt.gcf(), plot_frame, frames, interval=33.3)
    ani.save(save_path)

animate(cases[0].model, X_test, data.mouse[2], data.trail[2], n_preds=300, start_idx=3000, save_path='fig/linear_traj.mp4')
animate(cases[1].model, X_test, data.mouse[2], data.trail[2], n_preds=300, start_idx=3000, save_path='fig/lasso_traj.mp4')
animate(cases[2].model, X_test, data.mouse[2], data.trail[2], n_preds=300, start_idx=3000, save_path='fig/mlp_traj.mp4')


# <codecell>
start_idx=3000
n_preds=1
# model = LinearRegression()
# model.fit(X, y)

model = cases[2].model

xs = X_test[start_idx,:]
all_xs = [xs]

for i in range(n_preds):
    ys = model.predict(xs.reshape(1, -1)).flatten()
    mx, tx, my, ty = np.split(np.copy(xs), 4)

    mxn, txn, myn, tyn = np.split(X_test[start_idx+i+1],4)

    mx[:-1] = mx[1:]
    mx[-1] = ys[0]
    mx = mx - ys[0]

    my[:-1] = my[1:]
    my[-1] = ys[1]
    my = my - ys[1]

    dtx = txn[-1] - txn[-2]
    dty = tyn[-1] - tyn[-2]

    tx[:-1] = tx[1:]
    tx[-1] += dtx

    ty[:-1] = ty[1:]
    ty[-1] += dty

    plt.plot(mx, my)
    plt.plot(tx, ty)
    plt.ylim((-15, 15))

    xs = np.concatenate((mx, tx, my, ty))
    all_xs.append(xs)