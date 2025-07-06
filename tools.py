
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
def lorenz_96_ode(x, F):
    """
    Computes the right-hand side of the Lorenz-96 equations (vectorized).
    x: A numpy array of shape (N,) or (batch_size, N) representing the current state.
    F: The forcing parameter.
    """
    x_minus_1 = np.roll(x, 1, axis=-1) #y_{i-1} , use roll to handle periodic boundary conditions
    x_plus_1 = np.roll(x, -1, axis=-1) #y_{i+1}
    x_plus_2 = np.roll(x, -2, axis=-1) #y_{i+2}
    
    dxdt = (x_minus_1 - x_plus_2) * x_plus_1 - x + F
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


def run_closed_loop_and_plot(model, data_base, device, initial_state, num_simulation_steps, fig_dir_suffix):
    """
    Runs a closed-loop simulation and plots the PDFs of the state variables.
    """
    # Create a temporary coach instance for inference
    inference_coach = API.Coach(model, data_base, device)
    
    # Run closed-loop simulation
    simulated_trajectory = inference_coach.predict_closed_loop(initial_state, num_simulation_steps)
    print(f"Simulated trajectory shape: {simulated_trajectory.shape}")
    print("Closed-loop simulation complete.")

    # Plot PDFs
    print(f"\nGenerating PDF plots for simulation '{fig_dir_suffix}'...")
    plot_pdfs(
        ground_truth_trajectory=data_base.raw_trajectory,
        simulated_trajectory=simulated_trajectory,
        N_variables=API.config.N_LORENZ,
        fig_dir=Path(API.config.FIG_DIR) / fig_dir_suffix
    )
    print(f"PDF plots for simulation '{fig_dir_suffix}' generated and saved.")


def load_or_train_model(data_base, device, num_epochs):
    # Only PILSTMNet is supported
    model_class = API.PILSTMNet
    model_kwargs = {
        'input_dim': API.config.NX_LORENZ,
        'output_dim': API.config.N_LORENZ,
        'hidden_size': API.config.LSTM_HIDDEN_SIZE,
        'num_layers': API.config.LSTM_NUM_LAYERS
    }

    # Create a temporary model instance to get the model name for saving/loading paths
    temp_model = model_class(**model_kwargs)
    model_name = API.Coach(temp_model, data_base, device).get_name()
    model_path = Path(API.config.MODEL_SAVE_DIR) / f"{model_name}_epoch_{num_epochs}.pth"
    
    if model_path.exists():
        print(f"Loading model from {model_path}...")
        model, optimizer, scaler = API.Coach.load_model(model_class, model_path, device)
        # Update data_base's scaler with the loaded one
        data_base.scaler = scaler 
        print("Model loaded.")
    else:
        print("No saved model found. Starting training...")
        model = model_class(**model_kwargs)
        coach = API.Coach(
            dataBase = data_base, 
            device   = device,
            model    = model
        )
        coach.train(num_epochs=num_epochs)
        coach.save_model(num_epochs) # Save after training
        model = coach.model # Get the trained model instance
        scaler = data_base.scaler # Get the scaler from the data_base

    return model, scaler # Return model and the scaler used for it
