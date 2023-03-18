"""
Models for explaining trail tracking data

author: William Tong (wtong@g.harvard.edu)
date: March 6, 2023
"""

# <codecell>
import numpy as np
from torch.utils.data import Dataset

class MouseDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()




class Model:
    def fit(self):
        raise NotImplementedError('fit() not implemented')

    def predict(self):
        raise NotImplementedError('predict() not implemented')
