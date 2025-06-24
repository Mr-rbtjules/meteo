
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


###LORENZ SIMULATION###
cache = False
@njit(cache=cache)
def lorenz_96_ode(x, F):
    """
    Computes the right-hand side of the Lorenz-96 equations.
    x: A numpy array of shape (N,) representing the current state.
    F: The forcing parameter.
    """
    N = x.shape[0]
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i - 1 + N) % N] - x[(i + 2) % N]) * x[(i + 1) % N] - x[i] + F
    return dxdt

def simulate_lorenz_96(N, F, dt, num_steps, initial_state=None):
    """
    Simulates the Lorenz-96 system.
    N: Total number of system variables.
    F: Forcing parameter.
    dt: Time step for integration.
    num_steps: Number of simulation steps.
    initial_state: Optional initial state (numpy array of shape (N,)).
                   If None, a random initial state is generated.
    Returns: A numpy array of shape (num_steps, N) containing the trajectory.
    """
    if initial_state is None:
        np.random.seed(API.config.SEED_NP)
        x = np.random.rand(N) * 10 - 5 # Random initial state between -5 and 5
    else:
        x = initial_state.copy()

    trajectory = np.zeros((num_steps, N))
    trajectory[0] = x

    for i in range(1, num_steps):
        dxdt = lorenz_96_ode(x, F)
        x += dxdt * dt # Euler integration
        trajectory[i] = x
    return trajectory

def plot_pdfs(ground_truth_trajectory, simulated_trajectory, N_variables, fig_dir):
    """
    Computes and displays the probability-density functions (PDFs) of every state variable.
    ground_truth_trajectory: numpy array of shape (num_steps, N) for ground truth.
    simulated_trajectory: numpy array of shape (num_steps, N) for simulated data.
    N_variables: Total number of state variables (N_LORENZ).
    fig_dir: Directory to save the plots.
    """
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    print("Plotting PDFs for each state variable...")

    # Determine common bins and range for comparability
    # Combine data from both trajectories to find min/max for binning
    all_data = np.concatenate((ground_truth_trajectory, simulated_trajectory), axis=0)
    
    for i in range(N_variables):
        plt.figure(figsize=(8, 6))
        
        # Get data for the current variable
        gt_var_data = ground_truth_trajectory[:, i]
        sim_var_data = simulated_trajectory[:, i]

        # Determine min/max for this variable across both datasets
        min_val = np.min(all_data[:, i])
        max_val = np.max(all_data[:, i])
        
        # Create bins
        bins = np.linspace(min_val, max_val, 50) # 50 bins, adjust as needed

        # Compute histograms (PDFs)
        hist_gt, bin_edges = np.histogram(gt_var_data, bins=bins, density=True)
        hist_sim, _ = np.histogram(sim_var_data, bins=bins, density=True)

        # Plot
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(bin_centers, hist_gt, label='Ground Truth', color='blue', alpha=0.7)
        plt.plot(bin_centers, hist_sim, label='Model Simulation', color='red', alpha=0.7, linestyle='--')
        
        plt.title(f'PDF of State Variable y_{i}')
        plt.xlabel(f'y_{i} Value')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_filename = Path(fig_dir) / f"pdf_y{i}.png"
        plt.savefig(plot_filename)
        plt.close() # Close the figure to free memory
        print(f"Saved PDF plot for y_{i} to {plot_filename}")

    print("All PDF plots generated.")
