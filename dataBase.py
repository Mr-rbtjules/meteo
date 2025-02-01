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
from torch.utils.data import Dataset, DataLoader # type: ignore

class DataBase(Dataset):
    def __init__(self, origin_path, num_zones=100, n_similar=100):
        """
        Initialize the DataBase class, load data, normalize, split, and prepare DataLoaders.
        
        Parameters:
        - hdf5_dir (Path or str): Directory containing HDF5 trajectory files.
        - num_zones (int): Number of zone files to process.
        - n_similar (int): Number of trajectories per zone.
        - test_size (float): Fraction of data to reserve for testing.
        - random_state (int): Seed for reproducibility.
        - batch_size (int): Number of samples per batch in DataLoader.
        """
        super(DataBase, self).__init__()
        self.origin_path = Path(origin_path)
        self.num_zones = num_zones #another name for nb of files in the directory , 1 if uniform
        self.n_similar = n_similar
        self.batch_size = API.config.BATCH_SIZE
        
        # Load and prepare data
        self.time_var, self.space_coord = self.load_data()
        self.time_normalized, self.space_normalized, self.scaler_time, self.scaler_space = self.normalize_data()
        self.train_loader, self.test_loader = self.prepare_dataloaders()
    
    def load_data(self):
        """
        Load trajectories from HDF5 files and aggregate time and state data.
        
        Returns:
        - tuple: (time_var, space_coord)
            - time_var (numpy.ndarray): Time points, shape (N, 1)
            - space_coord (numpy.ndarray): States [x, y, z], shape (N, 3)
        """
        time_var = []
        space_coord = []
        for type_idx in range(1, self.num_zones + 1):
            file_path = self.origin_path / f"type_{type_idx:03d}.h5"
            if not file_path.exists():
                raise FileNotFoundError(f"HDF5 file not found: {file_path}")
            
            with h5py.File(file_path, 'r') as f:
                trajectories = f['trajectories'][:]  # Shape: (n_similar, 3, traj_steps)
                traj_steps = trajectories.shape[2]
                t_span = (0, 100)
                time_points = np.linspace(t_span[0], t_span[1], traj_steps)
                
                for sim in range(self.n_similar):
                    traj = trajectories[sim]
                    for t, state in zip(time_points, traj.T):
                        time_var.append(t)
                        space_coord.append(state)
        
        time_var = np.array(time_var).reshape(-1, 1)  # Shape: (N, 1)
        space_coord = np.array(space_coord)           # Shape: (N, 3)
        return time_var, space_coord
    
    def normalize_data(self):
        """
        Normalize time and state data using StandardScaler.
        
        Returns:
        - tuple: (time_normalized, space_normalized, scaler_time, scaler_space)
        """
        scaler_time = StandardScaler()
        scaler_space = StandardScaler()
        time_normalized = scaler_time.fit_transform(self.time_var)
        space_normalized = scaler_space.fit_transform(self.space_coord)
        return time_normalized, space_normalized, scaler_time, scaler_space
    
    def prepare_dataloaders(self):
        """
        Split data into training and testing sets and create DataLoader instances.
        
        Returns:
        - tuple: (train_loader, test_loader)
            - train_loader (DataLoader): DataLoader for training data.
            - test_loader (DataLoader): DataLoader for testing data.
        """
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            self.time_normalized, self.space_normalized,
            test_size=API.config.TEST_DATA_PROPORTION,
            random_state=API.config.SEED_SHUFFLE
        )
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        # Create TensorDatasets
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        print("Data normalized and split into training and testing sets.")
        return train_loader, test_loader
    
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.time_var)
    
    def __getitem__(self, idx):
        """
        Retrieve a single data point.
        
        Parameters:
        - idx (int): Index of the data point.
        
        Returns:
        - tuple: (input_time, target_state)
        """
        return self.time_normalized[idx], self.space_normalized[idx]
    
    def get_train_loader(self):
        """
        Get the DataLoader for training data.
        
        Returns:
        - DataLoader: Training DataLoader.
        """
        return self.train_loader
    
    def get_test_loader(self):
        """
        Get the DataLoader for testing data.
        
        Returns:
        - DataLoader: Testing DataLoader.
        """
        return self.test_loader
    @classmethod
    def createDbFromScratch(cls, output_dir,zones=100) -> None:
        sigma = 10.0
        rho = 28.0
        beta = 8.0 / 3.0

        # Define the ranges for x, y, z
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
        
        # Ensure exactly 100 zones
        if zone_centers.shape[0] != zones:
            raise ValueError(f"Expected {zones} zones, but got {zone_centers.shape[0]}.")
        
        # Parameters for trajectory generation
        t_span = (0, 100)  # Time range
        h = 1e-5  # Time step for dense evaluation
        n_similar = 100  # Number of similar trajectories per zone

        output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        
        # Initialize output directory
        start_time = time.time()
        

        ###precompilation#######################################################
        # Pre-compile by calling the function once
        dummy_ci = np.array([0.0, 0.0, 0.0])
        dummy_t_span = (0, 1)
        dummy_h = 1e-5
        API.tools.lorenz(0, dummy_ci, 10.0, 28.0, 8.0/3.0)
        API.tools.rk4_step(API.tools.lorenz, 0, dummy_ci, dummy_h, 10.0, 28.0, 8.0/3.0)
        API.tools.generate_trajectory(dummy_ci, dummy_t_span, dummy_h, 10.0, 28.0, 8.0/3.0)
        ##########################################################################


        zones = 100
        # Parallel processing: Generate and store each type in parallel
        Parallel(n_jobs=-1)(
            delayed(cls._generate_and_store_type)(
                type_idx=type_idx + 1,  # Type indices start at 1
                n_similar=n_similar,
                initial_state_base=zone_centers[type_idx],
                t_span=t_span,
                h=h,
                sigma=sigma,
                rho=rho,
                beta=beta,
                output_dir=output_dir
            )
            for type_idx in range(zones)  # type_idx from 0 to zones-1
        )

        # End timing
        end_time = time.time()
        print(f"Generated and stored all trajectories in {end_time - start_time:.2f} seconds")

    @classmethod
    def _generate_and_store_type(
            self, type_idx, n_similar, 
            initial_state_base, t_span,
            h, sigma, rho, beta, output_dir
        ):
        
        PERTURBATION = 0.05
        print("start job")
        file_name = os.path.join(output_dir, f"type_{type_idx:03d}.h5")
        t0, t_end = t_span
        save_interval = 1000  # Save 0.1% of points
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
            # Perturb the initial state slightly within the zone
            #perturbation_ranges = [1e-2, 5e-2, 1e-1]
            #for perturb in perturbation_ranges:
            #    perturbation = np.random.uniform(-perturb, perturb, size=3)
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

