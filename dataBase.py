import meteo_api as API
from pathlib import Path
import time
import numpy as np
from joblib import Parallel, delayed # type: ignore
import os
import h5py  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore





class DataBase:
   
    def __init__(
            self
    ) -> None:
        self.saved_trajectory_path = \
            Path(API.config.RAW_DATA_DIR) / "trajectories_dataset"
        print("db object created")

    def createDbFromScratch(self, output_dir,zones=100) -> None:
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
            delayed(API.tools.generate_and_store_type)(
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

    def generate_and_store_type(
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

    

    def load_trajectory(self, file_path, trajectory_index=0):
        """
        Load a specific trajectory from an HDF5 file.

        Parameters:
        - file_path: Path to the HDF5 file.
        - trajectory_index: Index of the trajectory to load (default is 0).

        Returns:
        - trajectory: NumPy array of shape (3, traj_steps) representing [x, y, z].
        - metadata: Dictionary containing metadata for the trajectory.
        """
        with h5py.File(file_path, 'r') as f:
            trajectories = f['trajectories'][:]  # Shape: (100, 3, traj_steps)
            selected_traj = trajectories[trajectory_index]  # Shape: (3, traj_steps)
            
            # Load metadata
            metadata = {}
            for key in f['metadata']:
                metadata[key] = f['metadata'][key][trajectory_index]
        
        return selected_traj, metadata
    


    def load_data(self, hdf5_dir, num_zones, n_similar):
        """
        Load trajectories from HDF5 files and prepare the dataset.
        
        Parameters:
        - hdf5_dir: Directory containing HDF5 trajectory files.
        - num_zones: Number of zone files.
        - n_similar: Number of trajectories per zone.
        
        Returns:
        - X: NumPy array of shape (N, 1) containing time points.
        - y: NumPy array of shape (N, 3) containing [x, y, z].
        """
        X = []
        y = []
        for type_idx in range(1, num_zones + 1):
            file_path = f"{hdf5_dir}/type_{type_idx:03d}.h5"
            with h5py.File(file_path, 'r') as f:
                trajectories = f['trajectories'][:]  # Shape: (n_similar, 3, traj_steps)
                traj_steps = trajectories.shape[2]
                t_span = (0, 100)  # Assuming same time span
                h = 1e-5
                time_points = np.linspace(t_span[0], t_span[1], traj_steps)
                for sim in range(n_similar):
                    traj = trajectories[sim]
                    for t, state in zip(time_points, traj.T):
                        X.append(t)
                        y.append(state)
        X = np.array(X).reshape(-1, 1)
        y = np.array(y)
        return X, y

    def final_load(self):
        # Load and prepare data
        hdf5_directory =  Path(API.config.RAW_DATA_DIR) / "trajectories_dataset"  # Replace with your path
        X, y = self.load_data(hdf5_directory, num_zones=1, n_similar=1)

        # Normalize the data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_normalized = scaler_X.fit_transform(X)
        y_normalized = scaler_y.fit_transform(y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)
        print("normalized and splited")