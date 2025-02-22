
from .config import SEED_NP
import numpy as np
from numba import njit  # type: ignore
import os
import h5py  # type: ignore
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt # type: ignore
import torch # type: ignore
import meteo_api as API

import matplotlib as mpl
mpl.rcParams['animation.convert_path'] = '/opt/homebrew/bin/convert'
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

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




def extract_param(model_name):
    """
    Given a model name string of the form:
      "cS{seq_len}_eD{embed_dim}_nL{num_layers}_nH{num_heads}_nT{num_tokens}_rl{resolution}_cF{context_fraction}_dS{load_size}"
    extract and return the parameters as a tuple:
      (seq_len, embed_dim, num_layers, num_heads, num_tokens, resolution, context_fraction, load_size)
    """
    parts = model_name.split('_')
    seq_len = int(parts[0][2:])
    embed_dim = int(parts[1][2:])
    num_layers = int(parts[2][2:])
    num_heads = int(parts[3][2:])
    num_tokens = int(parts[4][2:])
    resolution = int(parts[5][2:])
    context_fraction = float(parts[6][2:])
    load_size = float(parts[7][2:])
    return seq_len, embed_dim, num_layers, num_heads, num_tokens, resolution, context_fraction, load_size

def animate_trajectory(traj_data, title, save_path, interval=50, fps=None):
    """
    Creates and saves a simple 3D animation (GIF) of a trajectory using FuncAnimation.

    Parameters:
      - traj_data: numpy array of shape (T, 3) for (x, y, z) over time.
      - title:     Title for the plot.
      - save_path: Path to save the animation (e.g., "my_anim.gif" or "my_anim.mp4").
      - interval:  Delay between frames in milliseconds (for display; does not affect final FPS).
      - fps:       Frames per second when saving. If None, will use 1000 / interval.

    Note: If you give save_path a '.mp4' extension, Matplotlib will attempt to use
    ffmpeg or another writer to create an MP4. For a GIF, use '.gif'.
    """

    # If user did not specify fps, derive from interval (ms):
    if fps is None:
        fps = 1000.0 / interval

    # Create the figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    # Set up axis limits based on trajectory extents
    x_min, x_max = traj_data[:, 0].min(), traj_data[:, 0].max()
    y_min, y_max = traj_data[:, 1].min(), traj_data[:, 1].max()
    z_min, z_max = traj_data[:, 2].min(), traj_data[:, 2].max()

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Plot object that we will update. Here, a single 3D line.
    line, = ax.plot([], [], [], lw=2, c='blue')

    # Initialization function: called once before animation starts
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return (line,)

    # Update function: called for each frame
    def update(frame):
        # `frame` goes from 0 up to len(traj_data)-1
        x, y, z = traj_data[:frame+1, 0], traj_data[:frame+1, 1], traj_data[:frame+1, 2]
        line.set_data(x, y)
        line.set_3d_properties(z)
        return (line,)

    # Create the animation
    ani = FuncAnimation(
        fig,
        func=update,        # what to do each frame
        frames=len(traj_data), 
        init_func=init,     # initialization
        interval=interval,  # ms delay between frames
        blit=True
    )

    # Save the animation
    # If save_path ends with .gif, it uses pillow by default.
    # If it ends with .mp4, it will try ffmpeg or the configured writer.
    ani.save(save_path, fps=fps, writer='pillow' if save_path.endswith('.gif') else None)
    plt.close(fig)
    print(f"Animation saved to {save_path}")

def load_model(model, model_name, device):
    """
    Loads a saved model checkpoint into the given model instance.
    
    Parameters:
      - model: an instance of HybridNet.
      - model_name: the name string of the model (used to locate the checkpoint file).
      - device: torch.device to load the model on.
    
    Returns:
      The model with loaded weights, moved to device, and set to eval mode.
    """

    checkpoint_path = Path(API.config.MODELS_DIR) / (model_name + ".pth")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path} on {device}")
    return model


def make_animation(model_name, device, zone=99, traj_idx=0, interval=50):
    """
    Creates two animations:
      - Predicted trajectory: loads a saved model, runs predictions using the normalized context and full_time,
        denormalizes the predictions, and saves a GIF.
      - Ground truth trajectory: uses the denormalized full trajectory from the animation data.
      
    Both animations are automatically saved in the directory specified by API.config.FIG_DIR.
    
    Parameters:
      - model_name: the saved model name (matching the format from get_name()).
      - device: torch.device (e.g., torch.device("mps") or torch.device("cpu")).
      - zone: the zone number to select the trajectory.
      - traj_idx: index of the trajectory within that zone.
      - interval: frame interval (ms) for the animation.
    """
    # Extract model parameters from model_name.
    seq_len, embed_dim, num_layers, num_heads, num_tokens, resolution, context_fraction, load_size = extract_param(model_name)
    
    # Instantiate the database in animation mode.
    db = API.DataBase(context_fraction=context_fraction, context_tokens=seq_len,
                      load_size=load_size, resolution=resolution, animation=True)
    # Prepare animation data for the specified trajectory.
    db.prepare_data_for_animation(zone=zone, idx=traj_idx)
    anim_data = db.animation_data
    
    # Automatically construct save paths in FIG_DIR.
    fig_dir = Path(API.config.FIG_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_path_pred = str(fig_dir / (model_name + "_prediction.gif"))
    save_path_gt = str(fig_dir / (model_name + "_ground_truth.gif"))
    
    gt_full_state = anim_data["full_state"]
    animate_trajectory(gt_full_state, title=f"Ground Truth Trajectory (Zone {zone})", save_path=save_path_gt, interval=interval)


    # Instantiate a new HybridNet with matching configuration.
    model = API.HybridNet(seq_len=seq_len, input_dim=4, embed_dim=embed_dim,
                          num_layers=num_layers, num_heads=num_heads,
                          physnet_hidden=API.config.PHYSNET_HIDDEN, num_tokens=num_tokens)
    # Load model weights.
    model = load_model(model, model_name, device)

    # Get normalized context and full_time_norm for prediction.
    context = torch.tensor(anim_data["context_norm"], dtype=torch.float32, device=device).unsqueeze(0)  # (1, context_tokens, 4)
    full_time_norm = torch.tensor(anim_data["full_time_norm"], dtype=torch.float32, device=device).unsqueeze(0)  # (1, T_anim, 1)

    # Generate dynamic parameters from context.
    params = model.hypernet(context)

    # Instead of the while-loop:
    pred_full_norm = model.physnet(full_time_norm, params)  # shape (1, T_anim, 3) or (1, 1000, 3)
    if pred_full_norm.dim() == 2:
        pred_full_norm = pred_full_norm.unsqueeze(1)

    pred_full_norm = pred_full_norm.squeeze(0).detach().cpu().numpy()  # (T_anim, 3)
    pred_full = db.scaler_state.inverse_transform(pred_full_norm)

    animate_trajectory(pred_full, title=f"Predicted Trajectory (Zone {zone})", save_path=save_path_pred, interval=interval)
    
    
    
    
    
    
    
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




