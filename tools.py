
from .config import SEED_NP
import numpy as np
from numba import njit  # type: ignore
import os
import h5py  # type: ignore
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt # type: ignore
import torch # type: ignore

def batched_linear(x, weight, bias):
    """
    x = batch of t for first layer, do this function for each layer
    following by a relu fct
    Performs a batched linear transformation.
    x: Tensor of shape (B, N, in_features) or (B, in_features).
    weight: Tensor of shape (B, out_features, in_features).
    bias: Tensor of shape (B, out_features).
    
    Returns:
        Tensor of shape (B, N, out_features) (or (B, out_features) if N==1).
    """
    # If x is 2D, add a singleton time dimension.
    if x.dim() == 2:
        x = x.unsqueeze(1)  # (B, 1, in_features)
    # x: (B, N, in_features); weight: (B, out_features, in_features)
    out = torch.bmm(x, weight.transpose(1, 2)) + bias.unsqueeze(1)
    # Squeeze the time dimension if it is 1.
    if out.size(1) == 1:
        return out.squeeze(1)
    return out
 

###LORENZ SIMULATION###
cache = False
@njit(cache=cache)
def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

@njit(cache=cache)
def rk4_step(f, t, y, h, sigma, rho, beta):
    k1 = f(t, y, sigma, rho, beta)
    k2 = f(t + h / 2, y + h * k1 / 2, sigma, rho, beta)
    k3 = f(t + h / 2, y + h * k2 / 2, sigma, rho, beta)
    k4 = f(t + h, y + h * k3, sigma, rho, beta)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

@njit(cache=cache)
def generate_trajectory(ci, t_span, h, sigma, rho, beta, save_interval=1000):
    t0, t_end = t_span
    num_steps = int((t_end - t0) / h) + 1
    num_saved = num_steps // save_interval + 1
    trajectory = np.empty((3, num_saved))
    y = ci
    trajectory[:, 0] = y
    t = t0
    save_idx = 1
    for i in range(1, num_steps):
        y = rk4_step(lorenz, t, y, h, sigma, rho, beta)
        t += h
        if i % save_interval == 0:
            trajectory[:, save_idx] = y
            save_idx += 1
    return trajectory[:, :save_idx]


#uniform sample
def generate_zone_centers_uniform(x_range, y_range, z_range, num_zones=100, seed=SEED_NP):
    """
    Generate uniform zone centers within specified ranges using uniform random sampling.
    
    Parameters:
    - x_range: Tuple (min, max) for x
    - y_range: Tuple (min, max) for y
    - z_range: Tuple (min, max) for z
    - num_zones: Total number of zones to create (default 100)
    - seed: Random seed for reproducibility (default None)
    
    Returns:
    - NumPy array of shape (num_zones, 3) with zone centers
    """
    if seed is not None:
        np.random.seed(seed)
    
    x_centers = np.random.uniform(x_range[0], x_range[1], num_zones)
    y_centers = np.random.uniform(y_range[0], y_range[1], num_zones)
    z_centers = np.random.uniform(z_range[0], z_range[1], num_zones)
    
    zone_centers = np.vstack((x_centers, y_centers, z_centers)).T  # Shape: (num_zones, 3)
    
    return zone_centers


def plot_zone_centers(zone_centers):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = zone_centers[:, 0], zone_centers[:, 1], zone_centers[:, 2]
    ax.scatter(xs, ys, zs, c='blue', marker='o', alpha=0.6)
    ax.set_title("Uniformly Distributed Zone Centers")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def simulatePlotLorenz(sampling=0.1):

    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    # Parameters for trajectory generation
    t_span = (0, 100)  # Time range
    h = 1e-5  # Time step for dense evaluation
    ci = np.array([10.000000, 10.00000, 1.500000])
    total_steps = int((t_span[1] - t_span[0]) / h) + 1

    # Determine save_interval based on sampling_percentage
    # Ensure save_interval is at least 1
    save_interval = max(int(1 / (sampling / 100)), 1)

    print(f"Generating trajectory with save_interval={save_interval} ({sampling}%)...")
    trajectory = generate_trajectory(ci, t_span, h, sigma, rho, beta, save_interval=save_interval)
    print("Trajectory generation complete.")

    # Extract x, y, z coordinates
    x, y, z = trajectory

    # Create a 3D plot of the trajectory
    print("Plotting trajectory...")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x, y, z, lw=0.5, color='blue')
    ax.set_title(f"Lorenz System Trajectory CI: x = {ci[0]}  y = {ci[1]}  z = {ci[2]}", fontsize=16)
    ax.set_xlabel("X Axis", fontsize=12)
    ax.set_ylabel("Y Axis", fontsize=12)
    ax.set_zlabel("Z Axis", fontsize=12)

    # Optional: Enhance plot aesthetics
    ax.grid(True)
    ax.view_init(elev=30, azim=45)  # Adjust viewing angle

    plt.show()
    print("Plotting complete.")


def plotZones(zones=100) -> None:

    # Define the ranges for x, y, z
    x_range = (-20.0, 20.0)
    y_range = (-20.0, 30.0)
    z_range = (0.0, 40.0)
    
    # Generate zone centers using uniform sampling
    zone_centers = generate_zone_centers_uniform(x_range, y_range, z_range, num_zones=zones)
    
    # Ensure exactly 100 zones
    if zone_centers.shape[0] != zones:
        raise ValueError(f"Expected {zones} zones, but got {zone_centers.shape[0]}.")
    
    plot_zone_centers(zone_centers)


def plot_loaded_trajectory(trajectory, metadata=None, title=None, save_path=None):
    """
    Plot a single Lorenz trajectory in 3D.

    Parameters:
    - trajectory: NumPy array of shape (3, traj_steps).
    - metadata: (Optional) Dictionary containing metadata to include in the plot title.
    - title: (Optional) Custom title for the plot.
    - save_path: (Optional) Path to save the plot image.
    """
    x, y, z = trajectory
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(x, y, z, lw=0.5, color='blue')
    ax.set_title(title if title else "Lorenz Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    if metadata:
        sigma = metadata.get('sigma', None)
        rho = metadata.get('rho', None)
        beta = metadata.get('beta', None)
        init_x = metadata.get('initial_x', None)
        init_y = metadata.get('initial_y', None)
        init_z = metadata.get('initial_z', None)
        
        metadata_str = f"σ={sigma}, ρ={rho}, β={beta}\nInitial Conditions: x={init_x}, y={init_y}, z={init_z}"
        plt.figtext(0.15, 0.85, metadata_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def RemoveDirectoryContent(directory_path) -> None:
    """just a function to clean when we want to reset"""
    for root, dirs, files in os.walk(directory_path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            os.rmdir(dir_path)


"""
def filter_zones(zone_centers, singular_points, min_distance=5.0):
   
    filtered = []
    for center in zone_centers:
        too_close = False
        for sp in singular_points:
            distance = np.linalg.norm(np.array(center) - np.array(sp))
            if distance < min_distance:
                too_close = True
                break
        if not too_close:
            filtered.append(center)
    return filtered
"""