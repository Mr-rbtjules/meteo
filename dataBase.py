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


class TrajectoryDataset(Dataset):
    def __init__(self, reduced_trajectories):
        super(TrajectoryDataset, self).__init__()
        self.reduced = reduced_trajectories

    def __len__(self):
        return len(self.reduced)

    def __getitem__(self, idx):
        traj = self.reduced[idx]
        # Convert arrays to torch tensors.
        return {
            'context': torch.tensor(traj['context'], dtype=torch.float32),
            'grid_time': torch.tensor(traj['grid_time'], dtype=torch.float32),
            'grid_state': torch.tensor(traj['grid_state'], dtype=torch.float32),
            'inner_time': torch.tensor(traj['inner_time'], dtype=torch.float32),
            'inner_state': torch.tensor(traj['inner_state'], dtype=torch.float32),
            'full_time': torch.tensor(traj['full_time'], dtype=torch.float32),
            'full_state': torch.tensor(traj['full_state'], dtype=torch.float32),
            'zone': traj['zone']
        }

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


class DataBase:
    """
    The master database object.
    
    This object loads the raw data from HDF5 files, performs normalization,
    groups trajectories by zone, and then splits the dataset.
    
    We now add a reduction step so that for each trajectory we keep only the
    necessary points (from the first half), already split into context, grid, inner, and full.
    """
    def __init__(
            self, context_fraction=0.5, 
            context_tokens=1000, load_size=1,
            max_files=50, resolution=2
    ):
        self.load_size = load_size
        self.max_files = max_files
        self.origin_path = Path(API.config.RAW_DATA_DIR) / "trajectories_dataset/"
        self.context_fraction = context_fraction
        self.train_zone_fraction = 1 - API.config.TEST_DATA_PROPORTION
        self.within_zone_test_fraction = API.config.TEST_DATA_PROPORTION
        self.batch_size = API.config.BATCH_SIZE
        self.context_tokens = context_tokens
        self.resolution = resolution

        # Load trajectories (grouped by zone)
        self.zone2trajectories = self._load_trajectories()
        self.all_trajectories = []
        for zone, traj_list in self.zone2trajectories.items():
            self.all_trajectories.extend(traj_list)
        
        # Normalize trajectories
        self._normalize_trajectories()
        # Split the dataset (train/test)
        self._split_dataset()
        # Now, reduce each trajectory to the needed points.
        self._reduce_trajectories()

    def _load_trajectories(self):
        zone2traj = {}
        # Get all files matching pattern.
        h5_files = list(self.origin_path.glob("type_*.h5"))
        # Limit the number of files processed to max_files.
        h5_files = h5_files[:self.max_files]
        for file_path in h5_files:
            zone_id = int(file_path.stem.split("_")[1])
            with h5py.File(file_path, 'r') as f:
                zone_trajs = f['trajectories'][:int(len(f['trajectories'])*self.load_size)]
                traj_steps = zone_trajs.shape[2]
                t_span = (0, 100)
                traj_list = []
                for sim in range(zone_trajs.shape[0]):
                    traj = zone_trajs[sim].T  # shape: (traj_steps, 3)
                    traj_list.append({
                        'zone': zone_id,
                        'time': np.linspace(t_span[0], t_span[1], traj_steps).reshape(-1, 1),
                        'state': traj
                    })
                zone2traj[zone_id] = traj_list
        print("Data loaded from", len(h5_files), "files.")
        return zone2traj

    def _normalize_trajectories(self):
        # (Your original normalization code.)
        all_time = np.concatenate([traj['time'] for traj in self.all_trajectories], axis=0)
        all_state = np.concatenate([traj['state'] for traj in self.all_trajectories], axis=0)
        self.scaler_time = StandardScaler()
        self.scaler_state = StandardScaler()
        all_time_norm = self.scaler_time.fit_transform(all_time)
        all_state_norm = self.scaler_state.fit_transform(all_state)
        start_idx = 0
        for traj in self.all_trajectories:
            n_points = traj['time'].shape[0]
            traj['time'] = all_time_norm[start_idx:start_idx+n_points]
            traj['state'] = all_state_norm[start_idx:start_idx+n_points]
            start_idx += n_points
        print("Data normalized")

    def _split_dataset(self):
        # (Your original code splitting trajectories into train_list, test_within_list, test_unseen_list)
        all_zones = sorted(self.zone2trajectories.keys())
        n_train_zones = int(len(all_zones) * self.train_zone_fraction)
        train_zone_ids = all_zones[:n_train_zones]
        unseen_zone_ids = all_zones[n_train_zones:]
        self.train_zone_ids = train_zone_ids
        self.unseen_zone_ids = unseen_zone_ids

        train_list = []
        test_within_list = []
        for zone in train_zone_ids:
            trajs = self.zone2trajectories[zone]
            train_trajs, test_trajs = train_test_split(
                trajs, 
                test_size=self.within_zone_test_fraction,
                random_state=API.config.SEED_SHUFFLE)
            train_list.extend(train_trajs)
            test_within_list.extend(test_trajs)
        test_unseen_list = []
        for zone in unseen_zone_ids:
            test_unseen_list.extend(self.zone2trajectories[zone])
        self.train_list = train_list
        self.test_within_list = test_within_list
        self.test_unseen_list = test_unseen_list
        print("Data splitted")

    def _reduce_trajectories(self):
        """
        For each trajectory in all_trajectories, compute the reduced version and store it.
        Then, discard the full raw trajectories.
        """
        reduced_list = []
        # Iterate over every trajectory in the full dataset.
        for traj in self.all_trajectories:
            time_arr = traj['time']   # (traj_steps, 1) numpy array
            state_arr = traj['state'] # (traj_steps, 3) numpy array
            traj_steps = time_arr.shape[0]
            half_steps = int(traj_steps * self.context_fraction)
            if half_steps < self.context_tokens:
                continue  # or raise an error if desired
            
            # Use only the first half of the trajectory.
            time_half = time_arr[:half_steps]
            state_half = state_arr[:half_steps]
            
            # Define high_res_length = context_tokens * resolution_factor.
            high_res_length = self.context_tokens * self.resolution
            # Evenly sample indices from 0 to half_steps-1.
            indices_highres = np.linspace(0, half_steps - 1, num=high_res_length, dtype=int)
            highres_time = time_half[indices_highres]   # (high_res_length, 1)
            highres_state = state_half[indices_highres] # (high_res_length, 3)
            
            # Precompute the full high-res sample.
            full_time = highres_time.copy()
            full_state = highres_state.copy()
            
            # Context (for hypernetwork): take one every RESOLUTION_FACTOR-th point.
            indices_context = np.arange(0, high_res_length, self.resolution)
            context_time = highres_time[indices_context]   # (context_tokens, 1)
            context_state = highres_state[indices_context] # (context_tokens, 3)
            context = np.concatenate([context_state, context_time], axis=1)  # (context_tokens, 4)
            
            # Grid: same as context.
            grid_time = context_time.copy()
            grid_state = context_state.copy()
            
            # Inner: remaining points.
            all_indices = np.arange(high_res_length)
            indices_inner = np.setdiff1d(all_indices, indices_context)
            inner_time = highres_time[indices_inner]
            inner_state = highres_state[indices_inner]
            
            reduced_traj = {
                'zone': traj['zone'],
                'context': context,         # (context_tokens, 4)
                'grid_time': grid_time,     # (context_tokens, 1)
                'grid_state': grid_state,   # (context_tokens, 3)
                'inner_time': inner_time,   # (high_res_length - context_tokens, 1)
                'inner_state': inner_state, # (high_res_length - context_tokens, 3)
                'full_time': full_time,     # (high_res_length, 1)
                'full_state': full_state    # (high_res_length, 3)
            }
            reduced_list.append(reduced_traj)
        
        # After processing, discard the full trajectories to free memory.
        self.all_trajectories.clear()
        
        
        # Split reduced trajectories into train, test_within, and test_unseen lists.
        self.train_list = [traj for traj in reduced_list if traj['zone'] in self.train_zone_ids]
        self.test_within_list = [traj for traj in reduced_list if traj['zone'] in self.train_zone_ids]
        self.test_unseen_list = [traj for traj in reduced_list if traj['zone'] in self.unseen_zone_ids]
        print("Reduced trajectories prepared and full raw data discarded.")


    def get_train_loader(self, shuffle=True):
        train_dataset = TrajectoryDataset(self.train_list)
        #!! num of worker better low
        return DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=shuffle, num_workers=API.config.NUM_WORKERS
            )

    def get_test_within_loader(self, shuffle=False):
        test_within_dataset = TrajectoryDataset(self.test_within_list)
        return DataLoader(
            test_within_dataset, batch_size=self.batch_size,
            shuffle=shuffle, num_workers=API.config.NUM_WORKERS
            )

    def get_test_unseen_loader(self, shuffle=False):
        test_unseen_dataset = TrajectoryDataset(self.test_unseen_list)
        return DataLoader(
            test_unseen_dataset, batch_size=self.batch_size, 
            shuffle=shuffle, num_workers=API.config.NUM_WORKERS)