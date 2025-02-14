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

class TrajectoryDataset(Dataset):
    """
    A Dataset that splits each trajectory into a context portion and k query points.
    """
    def __init__(
            self,
            trajectories,
            resolution=10,
            context_fraction=0.5,
            max_token=1e3
        ):
        """
        Parameters:
          - trajectories: a list of trajectory dictionaries (each with 'zone', 'time', 'state')
          - k_query: number of query points to sample from the latter part of the trajectory
          - context_fraction: fraction of the trajectory to use as the context
        """
        super(TrajectoryDataset, self).__init__()
        self.trajectories = trajectories
        self.context_fraction = context_fraction
        self.max_token = max_token
        self.resolution = resolution

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj_sample = self.trajectories[idx]
        time_arr = traj_sample['time']    # full trajectory time, shape: (traj_steps, 1)
        state_arr = traj_sample['state']   # full trajectory state, shape: (traj_steps, 3)
        traj_steps = time_arr.shape[0]

        # Determine context length (for hypernetwork)
        context_len = int(self.context_fraction * traj_steps)
        if context_len >= traj_steps - 1:
            context_len = traj_steps - 2

        # Obtain context portion.
        context_time = time_arr[:context_len]
        context_state = state_arr[:context_len]
        full_context = torch.cat([
            torch.tensor(context_state, dtype=torch.float32), 
            torch.tensor(context_time, dtype=torch.float32)
        ], dim=-1)

        # Down-sample context if needed.
        if full_context.shape[0] > self.max_token:
            indices = torch.linspace(0, full_context.shape[0] - 1, steps=int(self.max_token)).long()
            context_sample = full_context[indices]
        else:
            context_sample = full_context

        # For available (grid) points, we use a down-sampled version.
        available_time = torch.tensor(time_arr[context_len:], dtype=torch.float32)
        available_state = torch.tensor(state_arr[context_len:], dtype=torch.float32)

        # Also return the full ground truth (if available) for evaluation.
        full_time = torch.tensor(time_arr, dtype=torch.float32)
        full_state = torch.tensor(state_arr, dtype=torch.float32)

        return {
            'context': context_sample,           # used by hypernetwork
            'available_time': available_time,      # grid points for training loss
            'available_state': available_state,    # grid ground truth for training
            'full_time': full_time,                # full resolution ground truth time
            'full_state': full_state,              # full resolution ground truth state
            'zone': traj_sample['zone']
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
    groups trajectories by zone, and then creates three dataset objects:
      - train_dataset: trajectories from training zones (using a fraction for training)
      - test_within_dataset: trajectories from training zones reserved for testing (initial condition sensitivity)
      - test_unseen_dataset: trajectories from unseen zones
    """
    def __init__(self, 
                 resolution=10, 
                 context_fraction=0.5,
                 max_token=1e3,
                 load_size=1
        ):
        self.load_size = load_size
        self.origin_path = Path(API.config.RAW_DATA_DIR) / "trajectories_dataset/"
        self.resolution = resolution
        self.context_fraction = context_fraction
        self.train_zone_fraction = 1 - API.config.TEST_DATA_PROPORTION
        self.within_zone_test_fraction = API.config.TEST_DATA_PROPORTION
        self.batch_size = API.config.BATCH_SIZE
        self.max_token = max_token

        # Load trajectories (grouped by zone)
        self.zone2trajectories = self._load_trajectories()
        #is a dict key = zone id, content = list of dict key = zone , state
        #comme ça mm qd shuffle on sait origine des zones
        #will point the same place as zone2traj
        self.all_trajectories = []
        for zone, traj_list in self.zone2trajectories.items():
            #traj_list is a dict with 
            self.all_trajectories.extend(traj_list)
        # Normalize globally
        self._normalize_trajectories()
        # Now split the dataset
        self._split_dataset()

    def _load_trajectories(self):
        """
        Load trajectories from HDF5 files.
        Each file (named like "type_001.h5") corresponds to one zone.
        Returns a dictionary mapping zone id to a list of trajectories.
        """
        zone2traj = {}
        h5_files = list(self.origin_path.glob("type_*.h5"))
        for file_path in h5_files:
            # Assume filename format "type_###.h5"
            zone_id = int(file_path.stem.split("_")[1])
            with h5py.File(file_path, 'r') as f:
                zone_trajs = f['trajectories'][:int(len(f['trajectories'])*self.load_size)]  # shape: (n_similar, 3, traj_steps)
                traj_steps = zone_trajs.shape[2]
                t_span = (0, 100)
                # Create time vector if not stored
                time_points = np.linspace(t_span[0], t_span[1], traj_steps).reshape(-1, 1)
                traj_list = []
                for sim in range(zone_trajs.shape[0]):
                    traj = zone_trajs[sim].T  # (traj_steps, 3)
                    traj_list.append({
                        'zone': zone_id, #redondant 
                        'time': time_points.copy(),
                        'state': traj
                    })
                zone2traj[zone_id] = traj_list
        print("data Loaded")
        return zone2traj

    def _normalize_trajectories(self):
        """
        Normalize all trajectories globally using StandardScaler.
        globally bc reference at same place
        """
        all_time = np.concatenate([traj['time'] for traj in self.all_trajectories], axis=0)
        all_state = np.concatenate([traj['state'] for traj in self.all_trajectories], axis=0)
        
        self.scaler_time = StandardScaler()
        self.scaler_state = StandardScaler()
        all_time_norm = self.scaler_time.fit_transform(all_time)
        all_state_norm = self.scaler_state.fit_transform(all_state)
        
        start_idx = 0
        for traj in self.all_trajectories:
            n_points = traj['time'].shape[0]
            traj['time'] = all_time_norm[start_idx:start_idx + n_points]
            traj['state'] = all_state_norm[start_idx:start_idx + n_points]
            start_idx += n_points

    def _split_dataset(self):
        """
        Create three lists of trajectories:
          - train_list: trajectories from training zones for training.
          - test_within_list: trajectories from training zones reserved as test samples (for initial condition sensitivity).
          - test_unseen_list: trajectories from zones not used for training.
        """
        all_zones = sorted(self.zone2trajectories.keys())
        n_train_zones = int(len(all_zones) * self.train_zone_fraction)
        train_zone_ids = all_zones[:n_train_zones]
        unseen_zone_ids = all_zones[n_train_zones:]

        self.train_zone_ids = train_zone_ids
        self.unseen_zone_ids = unseen_zone_ids

        train_list = []
        test_within_list = []
        # Split trajectories within each training zone.
        for zone in train_zone_ids:
            trajs = self.zone2trajectories[zone]
            train_trajs, test_trajs = train_test_split(
                trajs, 
                test_size=self.within_zone_test_fraction,
                random_state=API.config.SEED_SHUFFLE) #shuffle for split not zones
            train_list.extend(train_trajs)
            test_within_list.extend(test_trajs)

        test_unseen_list = []
        for zone in unseen_zone_ids:
            test_unseen_list.extend(self.zone2trajectories[zone])

        self.train_list = train_list
        self.test_within_list = test_within_list
        self.test_unseen_list = test_unseen_list

    def get_train_loader(self, shuffle=True):
        train_dataset = TrajectoryDataset(
            self.train_list, 
            resolution=self.resolution, 
            context_fraction=self.context_fraction,
            max_token=self.max_token
        )
        return DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=shuffle, num_workers=4
        )

    def get_test_within_loader(self, shuffle=False):
        test_within_dataset = TrajectoryDataset(
            self.test_within_list, 
            resolution=self.resolution, 
            context_fraction=self.context_fraction,
            max_token=self.max_token
        )
        return DataLoader(
            test_within_dataset, batch_size=self.batch_size, 
            shuffle=shuffle, num_workers=4
        )

    def get_test_unseen_loader(self, shuffle=False):
        test_unseen_dataset = TrajectoryDataset(
            self.test_unseen_list, 
            resolution=self.resolution, 
            context_fraction=self.context_fraction,
            max_token=self.max_token
        )
        return DataLoader(
            test_unseen_dataset, batch_size=self.batch_size,
            shuffle=shuffle, num_workers=4
        )

