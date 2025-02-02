

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

class DataBase(Dataset):
    def __init__(self, origin_path, num_zones=100, n_similar=100, context_fraction=0.5, train_zone_fraction=0.8):
        """
        Modified DataBase for dual usage by hypernetwork and physics network.
        
        Each sample is a full trajectory that is split into:
          - A context segment (e.g. first half of trajectory) for the hypernetwork.
          - A query time and corresponding state for the physics network.
        
        Additionally, the zones (files) are pre-split into training and testing sets.
        
        Parameters:
          - origin_path (str or Path): Directory containing HDF5 trajectory files.
          - num_zones (int): Total number of zone files.
          - n_similar (int): Number of trajectories per zone.
          - context_fraction (float): Fraction of a trajectory to use as context.
          - train_zone_fraction (float): Fraction of zones used for training (the rest for testing).
        """
        super(DataBase, self).__init__()
        self.origin_path = Path(origin_path)
        self.num_zones = num_zones
        self.n_similar = n_similar
        self.context_fraction = context_fraction
        self.batch_size = API.config.BATCH_SIZE
        
        # Load trajectories organized by zone
        self.all_trajectories = self.load_trajectories()  # List of dicts with keys: 'zone', 'time', 'state'
        
        # Normalize time and state globally
        self._normalize_trajectories()
        
        # Pre-split zones: reserve a fraction for training and the rest for testing
        self.train_trajectories, self.test_trajectories = self.split_by_zone(train_zone_fraction)
        
    def load_trajectories(self):
        """
        Load full trajectories from each HDF5 file.
        For each zone file, store each trajectory along with its zone id.
        
        Returns:
          A list of dictionaries, one per trajectory, each with keys:
            'zone': zone id (int)
            'time': numpy.ndarray of shape (traj_steps, 1)
            'state': numpy.ndarray of shape (traj_steps, 3)
        """
        trajectories_list = []
        # Assume zone files are named "type_001.h5", ..., "type_{num_zones:03d}.h5"
        for zone_idx in range(1, self.num_zones + 1):
            file_path = self.origin_path / f"type_{zone_idx:03d}.h5"
            if not file_path.exists():
                raise FileNotFoundError(f"HDF5 file not found: {file_path}")
            
            with h5py.File(file_path, 'r') as f:
                zone_trajs = f['trajectories'][:]  # Shape: (n_similar, 3, traj_steps)
                traj_steps = zone_trajs.shape[2]
                t_span = (0, 100)
                # Create a time vector (reshaped as (traj_steps, 1))
                time_points = np.linspace(t_span[0], t_span[1], traj_steps).reshape(-1, 1)
                
                # For each similar trajectory in this zone:
                for sim in range(self.n_similar):
                    traj = zone_trajs[sim]  # Shape: (3, traj_steps)
                    traj = traj.T  # Now shape: (traj_steps, 3)
                    trajectories_list.append({
                        'zone': zone_idx,
                        'time': time_points.copy(),  # (traj_steps, 1)
                        'state': traj               # (traj_steps, 3)
                    })
        return trajectories_list
    
    def _normalize_trajectories(self):
        """
        Normalize time and state for all trajectories.
        Uses StandardScaler on the concatenation of all data.
        Stores the scalers as attributes.
        """
        all_time = np.concatenate([traj['time'] for traj in self.all_trajectories], axis=0)
        all_state = np.concatenate([traj['state'] for traj in self.all_trajectories], axis=0)
        
        self.scaler_time = StandardScaler()
        self.scaler_state = StandardScaler()
        all_time_norm = self.scaler_time.fit_transform(all_time)
        all_state_norm = self.scaler_state.fit_transform(all_state)
        
        # Reassign normalized values back to each trajectory
        start_idx = 0
        for traj in self.all_trajectories:
            n_points = traj['time'].shape[0]
            traj['time'] = all_time_norm[start_idx:start_idx + n_points]
            traj['state'] = all_state_norm[start_idx:start_idx + n_points]
            start_idx += n_points
    
    def split_by_zone(self, train_zone_fraction):
        """
        Split the trajectories into training and testing sets by zone.
        For example, if train_zone_fraction is 0.8 and there are 100 zones,
        use trajectories from zones 1-80 for training and 81-100 for testing.
        
        Returns:
          (train_trajectories, test_trajectories): two lists of trajectory dicts.
        """
        # First, group trajectories by zone
        zones = {}
        for traj in self.all_trajectories:
            zone = traj['zone']
            if zone not in zones:
                zones[zone] = []
            zones[zone].append(traj)
        
        sorted_zones = sorted(zones.keys())
        n_train_zones = int(len(sorted_zones) * train_zone_fraction)
        train_zone_ids = sorted_zones[:n_train_zones]
        test_zone_ids = sorted_zones[n_train_zones:]
        
        train_trajectories = []
        test_trajectories = []
        for zone_id, trajs in zones.items():
            if zone_id in train_zone_ids:
                train_trajectories.extend(trajs)
            else:
                test_trajectories.extend(trajs)
        
        print(f"Train zones: {train_zone_ids}")
        print(f"Test zones: {test_zone_ids}")
        return train_trajectories, test_trajectories
    
    def __len__(self):
        """
        Return the number of training trajectory samples.
        (If using for training, we index into the training set.)
        """
        return len(self.train_trajectories)
    
    def __getitem__(self, idx):
        """
        For a given trajectory sample (from the training set), split it into:
          - context: the first part of the trajectory (e.g., first context_fraction)
          - query: a randomly chosen time point (and corresponding state) from the remainder.
        
        Returns a dictionary with:
          'context_time': Tensor, shape (context_steps, 1)
          'context_state': Tensor, shape (context_steps, 3)
          'query_time': Tensor, scalar (or shape (1,))
          'query_state': Tensor, shape (3,)
          'zone': the zone id (optional, may be useful for analysis)
        """
        traj_sample = self.train_trajectories[idx]
        time_arr = traj_sample['time']  # (traj_steps, 1)
        state_arr = traj_sample['state']  # (traj_steps, 3)
        traj_steps = time_arr.shape[0]
        
        # Determine context length
        context_len = int(self.context_fraction * traj_steps)
        # Ensure there's at least one query point
        if context_len >= traj_steps - 1:
            context_len = traj_steps - 2
        
        # Context: first context_len points
        context_time = time_arr[:context_len]
        context_state = state_arr[:context_len]
        
        # Query: randomly select one point from the remainder
        query_idx = random.randint(context_len, traj_steps - 1)
        query_time = time_arr[query_idx]   # shape (1,)
        query_state = state_arr[query_idx]  # shape (3,)
        
        return {
            'context_time': torch.tensor(context_time, dtype=torch.float32),
            'context_state': torch.tensor(context_state, dtype=torch.float32),
            'query_time': torch.tensor(query_time, dtype=torch.float32).squeeze(),  # scalar or (1,)
            'query_state': torch.tensor(query_state, dtype=torch.float32),
            'zone': traj_sample['zone']
        }
    
    def get_train_loader(self):
        """
        Create a DataLoader for training trajectories.
        """
        return DataLoader(self, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def get_test_loader(self):
        """
        Create a DataLoader for testing trajectories (from unseen zones).
        Here, we define a simple Dataset that returns the full trajectory.
        """
        test_dataset = TrajectoryTestDataset(self.test_trajectories)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
    

class TrajectoryTestDataset(Dataset):
    """
    A simple dataset to return the full trajectory for test samples.
    You can use the full trajectory to generate context and queries as needed,
    or evaluate continuous predictions.
    """
    def __init__(self, trajectories):
        self.trajectories = trajectories
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj_sample = self.trajectories[idx]
        return {
            'time': torch.tensor(traj_sample['time'], dtype=torch.float32),
            'state': torch.tensor(traj_sample['state'], dtype=torch.float32),
            'zone': traj_sample['zone']
        }
    
    def get_zone_ids(self):
        return list({traj['zone'] for traj in self.trajectories})


    @classmethod
    def createDbFromScratch(cls, output_dir, zones=100) -> None:
        """
        This method is kept as in your original implementation.
        Generates the trajectories and stores them into HDF5 files.
        """
        sigma = 10.0
        rho = 28.0
        beta = 8.0 / 3.0

        # Define ranges for x, y, z
        x_range = (-20.0, 20.0)
        y_range = (-20.0, 30.0)
        z_range = (0.0, 40.0)
        
        # Generate zone centers using uniform sampling
        zone_centers = API.tools.generate_zone_centers_uniform(
            x_range,
            y_range, 
            z_range,
            num_zones=zones
        )
        
        if zone_centers.shape[0] != zones:
            raise ValueError(f"Expected {zones} zones, but got {zone_centers.shape[0]}.")
        
        t_span = (0, 100)
        h = 1e-5
        n_similar = 100

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pre-compilation step as before
        dummy_ci = np.array([0.0, 0.0, 0.0])
        dummy_t_span = (0, 1)
        dummy_h = 1e-5
        API.tools.lorenz(0, dummy_ci, 10.0, 28.0, 8.0/3.0)
        API.tools.rk4_step(API.tools.lorenz, 0, dummy_ci, dummy_h, 10.0, 28.0, 8.0/3.0)
        API.tools.generate_trajectory(dummy_ci, dummy_t_span, dummy_h, 10.0, 28.0, 8.0/3.0)

        start_time = time.time()
        Parallel(n_jobs=-1)(
            delayed(cls._generate_and_store_type)(
                type_idx=type_idx + 1,
                n_similar=n_similar,
                initial_state_base=zone_centers[type_idx],
                t_span=t_span,
                h=h,
                sigma=sigma,
                rho=rho,
                beta=beta,
                output_dir=output_dir
            )
            for type_idx in range(zones)
        )
        end_time = time.time()
        print(f"Generated and stored all trajectories in {end_time - start_time:.2f} seconds")

    @classmethod
    def _generate_and_store_type(
            cls, type_idx, n_similar, 
            initial_state_base, t_span,
            h, sigma, rho, beta, output_dir
        ):
        PERTURBATION = 0.05
        print("start job")
        file_name = os.path.join(output_dir, f"type_{type_idx:03d}.h5")
        t0, t_end = t_span
        save_interval = 1000
        traj_steps = int((t_end - t0) / h) // save_interval + 1
        
        trajectories = np.empty((n_similar, 3, traj_steps), dtype=np.float32)
        metadata = {
            'sigma': np.full(n_similar, sigma, dtype=np.float32),
            'rho': np.full(n_similar, rho, dtype=np.float32),
            'beta': np.full(n_similar, beta, dtype=np.float32),
            'initial_x': np.empty(n_similar, dtype=np.float32),
            'initial_y': np.empty(n_similar, dtype=np.float32),
            'initial_z': np.empty(n_similar, dtype=np.float32)
        }
        
        for sim in range(n_similar):
            perturbation = np.random.uniform(-PERTURBATION, PERTURBATION, size=3)
            ci = initial_state_base + perturbation
            traj = API.tools.generate_trajectory(ci, t_span, h, sigma, rho, beta, save_interval)
            trajectories[sim] = traj
            metadata['initial_x'][sim] = ci[0]
            metadata['initial_y'][sim] = ci[1]
            metadata['initial_z'][sim] = ci[2]
        
        with h5py.File(file_name, 'w') as f:
            f.create_dataset('trajectories', data=trajectories, compression=None)
            for key, value in metadata.items():
                f.create_dataset(f"metadata/{key}", data=value, compression=None)
        
        print(f"Stored type {type_idx} with {n_similar} trajectories in {file_name}")