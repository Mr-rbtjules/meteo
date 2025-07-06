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


# Define the indices for observed variables (y1, y3, y5, y7, y9) as a module-level constant
# This corresponds to 0-indexed variables y_0, y_2, y_4, y_6, y_8
OBS_IDXS = np.arange(API.config.N_LORENZ)[::2][:API.config.NX_LORENZ]

class LSTMTrajectoryDataset(Dataset): # For LSTM
    def __init__(self, full_trajectory, raw_trajectory, indices):
        super(LSTMTrajectoryDataset, self).__init__()
        self.full_trajectory = full_trajectory # Normalized data
        self.raw_trajectory = raw_trajectory # Unnormalized data (for physics loss)
        self.indices = indices # Indices of starting points for segments

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the starting index for this segment
        segment_start_idx = self.indices[idx]

        # Extract the segment of observed inputs (x_segment)
        # This is a sequence of length SEGMENT_LENGTH, where each element is an x(t)
        # Shape: (SEGMENT_LENGTH, NX_LORENZ)
        x_segment = self.full_trajectory[segment_start_idx : segment_start_idx + API.config.SEGMENT_LENGTH, OBS_IDXS]

        # Extract the corresponding target segment (y_segment_targets)
        # This is y(t+1) for each x(t) in x_segment
        # Shape: (SEGMENT_LENGTH, N_LORENZ)
        y_segment_targets = self.full_trajectory[segment_start_idx + 1 : segment_start_idx + API.config.SEGMENT_LENGTH + 1, :]

        # Extract the corresponding current full states (y_segment_current_full_states)
        # This is y(t) for each x(t) in x_segment, needed for physics loss
        # Shape: (SEGMENT_LENGTH, N_LORENZ)
        y_segment_current_full_states = self.full_trajectory[segment_start_idx : segment_start_idx + API.config.SEGMENT_LENGTH, :]

        return (
            torch.tensor(x_segment, dtype=torch.float32),
            torch.tensor(y_segment_targets, dtype=torch.float32),
            torch.tensor(y_segment_current_full_states, dtype=torch.float32)
        )


class DataBase:
    """
    The master database object.
    
    This object loads the raw data from HDF5 files, performs normalization,
    and then splits the dataset into training and validation data pairs.
    """
    def __init__(self):
        self.data_file_path = Path(API.config.RAW_DATA_DIR) / "lorenz_processed_data.h5"
        
        # Load or generate processed data
        self.trajectory, self.raw_trajectory, self.scaler, self.train_lstm_segment_indices, self.val_lstm_segment_indices = self._load_or_generate_processed_data()
        
        print(f"DataBase initialized. Data loaded/generated and normalized.")
        print(f"Total trajectory length: {len(self.trajectory)}")
        print(f"Number of training LSTM segments: {len(self.train_lstm_segment_indices)}")
        print(f"Number of validation LSTM segments: {len(self.val_lstm_segment_indices)}")

    def _load_or_generate_processed_data(self):
        if self.data_file_path.exists():
            print(f"Loading processed Lorenz-96 data from {self.data_file_path}")
            with h5py.File(self.data_file_path, 'r') as f:
                trajectory = f['normalized_trajectory'][:]
                raw_trajectory = f['raw_trajectory'][:]
                scaler_mean = f['scaler_mean'][:]
                scaler_scale = f['scaler_scale'][:]
                train_indices = f['train_lstm_segment_indices'][:]
                val_indices = f['val_lstm_segment_indices'][:]
            
            scaler = StandardScaler()
            scaler.mean_ = scaler_mean
            scaler.scale_ = scaler_scale
            scaler.n_features_in_ = API.config.N_LORENZ # Manually set n_features_in_

            return trajectory, raw_trajectory, scaler, train_indices, val_indices
        else:
            print(f"Generating Lorenz-96 data, normalizing, and saving to {self.data_file_path}")
            self.data_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate raw data
            raw_trajectory = API.tools.simulate_lorenz_96(
                N=API.config.N_LORENZ,
                F=API.config.F_LORENZ,
                dt=API.config.DT_LORENZ,
                num_steps=API.config.NUM_STEPS_LORENZ
            )
            
            # Normalize data
            scaler = StandardScaler()
            normalized_trajectory = scaler.fit_transform(raw_trajectory.reshape(-1, API.config.N_LORENZ)).reshape(raw_trajectory.shape)
            
            # Prepare LSTM segment indices
            num_lstm_segments = len(normalized_trajectory) - (API.config.SEGMENT_LENGTH + 1)
            all_lstm_segment_start_indices = np.arange(num_lstm_segments)
            # Deterministic train/validation split
            rng = np.random.RandomState(API.config.SEED_SHUFFLE)
            train_lstm_segment_indices, val_lstm_segment_indices = train_test_split(
                all_lstm_segment_start_indices,
                test_size=API.config.TEST_DATA_PROPORTION,
                random_state=rng,
                shuffle=True,
            )
            
            # Save to HDF5
            with h5py.File(self.data_file_path, 'w') as f:
                f.create_dataset('normalized_trajectory', data=normalized_trajectory)
                f.create_dataset('raw_trajectory', data=raw_trajectory)
                f.create_dataset('scaler_mean', data=scaler.mean_)
                f.create_dataset('scaler_scale', data=scaler.scale_)
                f.create_dataset('train_lstm_segment_indices', data=train_lstm_segment_indices)
                f.create_dataset('val_lstm_segment_indices', data=val_lstm_segment_indices)
            
            return normalized_trajectory, raw_trajectory, scaler, train_lstm_segment_indices, val_lstm_segment_indices
