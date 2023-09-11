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
from scipy.linalg import block_diag
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler

from model import MlpModel
from preprocess import segment

data = pd.read_pickle('data/df.pkl')

train_idxs = np.arange(len(data) - 1)
test_idx = len(data) - 1

segs = [segment(data.mouse.iloc[i], data.trail[i], flatten=True) for i in train_idxs]
train_Xs, train_ys = zip(*segs)

X = np.concatenate(train_Xs)
y = np.concatenate(train_ys)

scalar_x = StandardScaler().fit(X)
scalar_y = StandardScaler().fit(y)

X = scalar_x.transform(X)
y = scalar_y.transform(y)

X_test, y_test = segment(data.mouse[test_idx], data.trail[test_idx], flatten=True)
X_test = scalar_x.transform(X_test)
y_test = scalar_y.transform(y_test)

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
plt.savefig('fig/1d_feats_comparison_wide.png')

# <codecell>
def plot_animation(model, save_path):
    plot_idx = len(data) - 1  # corresponds to the test index

    X_plot, y_plot = segment(data.mouse[plot_idx], data.trail[plot_idx], flatten=True)
    X_plot_sc = scalar_x.transform(X_plot)
    y_plot_pred_sc = model.predict(X_plot_sc)
    y_plot_pred = scalar_y.inverse_transform(y_plot_pred_sc)

    X_plot = X_plot.reshape(X_plot.shape[0], 20, -1)

    def plot_frame(t_idx):
        plt.clf()
        X, y, y_pred = X_plot[t_idx], y_plot[t_idx], y_plot_pred[t_idx]

        rx, ry, ux, uy, lx, ly, dwx, dwy, tx_dist, ty_dist = X[10:, -1]

        if ux == uy == 0:
            theta = 0
        else:
            theta = np.arctan(ux / uy)

        if uy < 0:
            theta += np.pi

        rot_mat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        rot_mat_block = block_diag(*(10 * [rot_mat]))
        X = rot_mat_block @ X
        y = (rot_mat @ y.reshape(-1, 1)).flatten()
        y_pred = (rot_mat @ y_pred.reshape(-1, 1)).flatten()


        mx = X[0, :]
        my = X[1, :]

        px = mx[-1]
        py = my[-1]

        rx, ry, ux, uy, lx, ly, dwx, dwy, tx_dist, ty_dist = X[10:, -1]

        plt.plot(mx, my)

        plt.arrow(px, py, y[0]-px, y[1]-py, head_width=0.5, head_length=0.5, color='red')
        plt.arrow(px, py, y_pred[0]-px, y_pred[1]-py, head_width=0.5, head_length=0.5, color='purple')

        plt.arrow(px, py, rx, ry, color='gray', alpha=0.5)
        plt.arrow(px, py, ux, uy, color='gray', alpha=0.5)
        plt.arrow(px, py, lx, ly, color='gray', alpha=0.5)
        plt.arrow(px, py, dwx, dwy, color='gray', alpha=0.5)

        plt.arrow(px, py, tx_dist, ty_dist, color='orange')
        plt.xticks([])
        plt.yticks([])

    ani = FuncAnimation(plt.gcf(), plot_frame, np.arange(X_plot.shape[0]), interval=330)
    ani.save(save_path)

# TODO: visualize tracks more closely; set up tuning
plot_animation(cases[0].model, 'fig/linear_feats_traj_wide.mp4')
plot_animation(cases[1].model, 'fig/lasso_feats_traj_wide.mp4')
plot_animation(cases[2].model, 'fig/mlp_feats_traj_wide.mp4')
