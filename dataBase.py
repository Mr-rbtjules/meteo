import meteo_api as API
from pathlib import Path
import time
import numpy as np
from joblib import Parallel, delayed  # type: ignore
import os
import h5py  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
import torch  # type: ignore
from torch.utils.data import Dataset, DataLoader  # type: ignore
import random

"""
possibilités : 
-batch size
-full dataloading
-too big precision
"""


def worker_init_fn(worker_id):
    base_seed = API.config.SEED_TORCH  # make sure you set this in your config
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


class TrajectoryDataset(Dataset):
    def __init__(self, reduced_trajectories):
        super(TrajectoryDataset, self).__init__()
        self.reduced = reduced_trajectories

    def __len__(self):
        return len(self.reduced)

    def __getitem__(self, idx):
        return None


class DataBase:
    """
    The master database object.
    
    This object loads the raw data from HDF5 files, performs normalization,
    groups trajectories by zone, and then splits the dataset.
    
    We now add a reduction step so that for each trajectory we keep only the
    necessary points (from the first half), already split into context, grid, inner, and full.
    """
    def __init__(
            self
    ):
        pass

    def _normalize_trajectories(self):
       return None

    