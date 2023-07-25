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
from sklearn.preprocessing import StandardScaler

from model import MlpModel
from preprocess import segment

data = pd.read_pickle('data/df.pkl')

train_idxs = np.arange(4)
test_idx = 4

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
    # Case(name='MLP', model=MlpModel(), args={'X_test': X_test, 'y_test': y_test}),
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
plt.savefig('fig/1d_feats_comparison.png')

# <codecell>
plot_idx = 4
model = cases[0].model

X_plot, y_plot = segment(data.mouse[plot_idx], data.trail[plot_idx], flatten=True)
X_plot_sc = scalar_x.transform(X_plot)
y_plot_pred_sc = model.predict(X_plot_sc)
y_plot_pred = scalar_y.inverse_transform(y_plot_pred_sc)

X_plot = X_plot.reshape(X_plot.shape[0], 20, -1)

# TODO: correct orientation and animate <-- STOPPED HERE
t_idx = 463
X, y, y_pred = X_plot[t_idx], y_plot[t_idx], y_plot_pred[t_idx]

mx = X[0, :]
my = X[1, :]

px = mx[-1]
py = my[-1]

rx, ry, ux, uy, lx, ly, dwx, dwy, tx_dist, ty_dist = X[10:, -1]
# llx, lly, lrx, lry, ulx, uly, urx, ury = X[idx, 16:,-1]
# print(llx)

dx = X[0,-1] - X[8,-1]
dy = X[1,-1] - X[9,-1]

plt.plot(mx, my)

plt.arrow(px, py, y[0]-px, y[1]-py, head_width=0.5, head_length=0.5, color='red')
plt.arrow(px, py, y_pred[0]-px, y_pred[1]-py, head_width=0.5, head_length=0.5, color='purple')

plt.arrow(px, py, rx, ry, color='gray', alpha=0.5)
plt.arrow(px, py, ux, uy, color='gray', alpha=0.5)
plt.arrow(px, py, lx, ly, color='gray', alpha=0.5)
plt.arrow(px, py, dwx, dwy, color='gray', alpha=0.5)

plt.arrow(px, py, tx_dist, ty_dist, color='orange')



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