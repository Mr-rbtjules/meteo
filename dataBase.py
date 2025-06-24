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
        # Calculate the number of full segments that can be extracted.
        # Each segment needs to contain SEQUENCE_LENGTH + SEGMENT_LENGTH - 1 points
        # to provide SEGMENT_LENGTH (input_sequence, target, current_full_state) triplets.
        # The last possible start index for a sequence is len(self.reduced) - API.config.SEQUENCE_LENGTH - 1.
        # So, the number of segments is based on how many times SEGMENT_LENGTH fits into the available data
        # after accounting for the SEQUENCE_LENGTH for the last prediction.
        return (len(self.reduced) - API.config.SEQUENCE_LENGTH) // API.config.SEGMENT_LENGTH

    def __getitem__(self, idx):
        # Calculate the starting index for this segment in the full trajectory
        segment_start_idx = idx * API.config.SEGMENT_LENGTH

        # Initialize lists to hold the batch of sequences and targets for this segment
        segment_x_seqs = []
        segment_y_nexts = []
        segment_y_current_full_states = []

        # Iterate through the segment to create individual (input, target, current_state) triplets
        for i in range(API.config.SEGMENT_LENGTH):
            current_data_idx = segment_start_idx + i

            # Input sequence: observed variables x(t) for SEQUENCE_LENGTH timesteps
            # Shape: (SEQUENCE_LENGTH, NX_LORENZ)
            x_seq = self.reduced[current_data_idx : current_data_idx + API.config.SEQUENCE_LENGTH, :API.config.NX_LORENZ]
            
            # Target: full state y(t+1) for the timestep immediately following the input sequence
            # Shape: (N_LORENZ,)
            y_next = self.reduced[current_data_idx + API.config.SEQUENCE_LENGTH, :]

            # The full state at the last timestep of the input sequence (y(t_i))
            # This is needed for the physics-informed loss to calculate the finite difference.
            # Shape: (N_LORENZ,)
            y_current_full_state = self.reduced[current_data_idx + API.config.SEQUENCE_LENGTH - 1, :]

            segment_x_seqs.append(x_seq)
            segment_y_nexts.append(y_next)
            segment_y_current_full_states.append(y_current_full_state)

        # Stack the lists to create tensors for the entire segment
        # Resulting shapes:
        # segment_x_seqs_tensor: (SEGMENT_LENGTH, SEQUENCE_LENGTH, NX_LORENZ)
        # segment_y_nexts_tensor: (SEGMENT_LENGTH, N_LORENZ)
        # segment_y_current_full_states_tensor: (SEGMENT_LENGTH, N_LORENZ)
        segment_x_seqs_tensor = torch.tensor(np.array(segment_x_seqs), dtype=torch.float32)
        segment_y_nexts_tensor = torch.tensor(np.array(segment_y_nexts), dtype=torch.float32)
        segment_y_current_full_states_tensor = torch.tensor(np.array(segment_y_current_full_states), dtype=torch.float32)

        return segment_x_seqs_tensor, segment_y_nexts_tensor, segment_y_current_full_states_tensor


class DataBase:
    """
    The master database object.
    
    This object loads the raw data from HDF5 files, performs normalization,
    groups trajectories by zone, and then splits the dataset.
    
    We now add a reduction step so that for each trajectory we keep only the
    necessary points (from the first half), already split into context, grid, inner, and full.
    """
    def __init__(self):
        self.raw_data_path = Path(API.config.RAW_DATA_DIR) / "lorenz_trajectory.h5"
        self.trajectory = self._load_or_generate_data()
        self.scaler = StandardScaler()
        self.trajectory = self._normalize_trajectories(self.trajectory)
        
        # For simplicity, we'll use the entire trajectory as 'reduced' for now.
        # In a real scenario, you might split this into train/test or apply further reduction.
        self.reduced = self.trajectory 

    def _load_or_generate_data(self):
        if self.raw_data_path.exists():
            print(f"Loading Lorenz-96 data from {self.raw_data_path}")
            with h5py.File(self.raw_data_path, 'r') as f:
                trajectory = f['trajectory'][:]
        else:
            print(f"Generating Lorenz-96 data and saving to {self.raw_data_path}")
            self.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory = API.tools.simulate_lorenz_96(
                N=API.config.N_LORENZ,
                F=API.config.F_LORENZ,
                dt=API.config.DT_LORENZ,
                num_steps=API.config.NUM_STEPS_LORENZ
            )
            with h5py.File(self.raw_data_path, 'w') as f:
                f.create_dataset('trajectory', data=trajectory)
        return trajectory

    def _normalize_trajectories(self, trajectory):
        # Fit scaler on the entire trajectory and transform
        # Reshape for scaler: (num_samples, num_features)
        normalized_trajectory = self.scaler.fit_transform(trajectory.reshape(-1, API.config.N_LORENZ))
        return normalized_trajectory.reshape(trajectory.shape) # Reshape back to original (num_steps, N)
