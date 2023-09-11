"""
Plot example tracks of mice trails
"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_pickle('data/df.pkl')
df.head()

# <codecell>
mouse_idx = 2
start_idx = 300
window_size = 200

size_fac = window_size / 35

plt.gcf().set_size_inches(3 * size_fac, 3)

mouse = df.iloc[mouse_idx]['mouse']
trail = df.iloc[mouse_idx]['trail']

trail.shape

plt.plot(trail[0,:], trail[1,:], color='red')
plt.plot(mouse[0,:], mouse[1,:], color='black', alpha=0.5)
plt.xlim(start_idx, start_idx + window_size)
