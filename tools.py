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
    # Roll to obtain periodic neighbours
    x_im1 = np.roll(x, 1, axis=-1)   # y_{i-1}
    x_ip1 = np.roll(x, -1, axis=-1)  # y_{i+1}
    x_im2 = np.roll(x, 2, axis=-1)   # y_{i-2}

    # Lorenz‑96 RHS: (y_{i+1} − y_{i-2}) * y_{i-1} − y_i + F
    dxdt = (x_ip1 - x_im2) * x_im1 - x + F
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


def run_closed_loop_and_plot(model, data_base, device, initial_state, num_simulation_steps, model_name_for_dir):
    """
    Runs a closed-loop simulation and plots the PDFs of the state variables.
    """
    # Select the appropriate coach class based on the model type
    if model.model_type == 'PILSTM':
        coach_class = API.Coach
    elif model.model_type == 'Transformer':
        coach_class = API.TFCoach
    else:
        raise ValueError(f"Unknown model type: {model.model_type}")

    # Create a temporary coach instance for inference (do not pass total_target_epochs to prevent SummaryWriter initialization)
    inference_coach = coach_class(model, data_base, device, total_target_epochs=None)
    
    # Run closed-loop simulation
    simulated_trajectory = inference_coach.predict_closed_loop(initial_state, num_simulation_steps)
    print(f"Simulated trajectory shape: {simulated_trajectory.shape}")
    print("Closed-loop simulation complete.")

    # Plot PDFs
    print(f"\nGenerating PDF plots for simulation '{model_name_for_dir}'...")
    plot_pdfs(
        ground_truth_trajectory=data_base.raw_trajectory,
        simulated_trajectory=simulated_trajectory,
        N_variables=API.config.N_LORENZ,
        fig_dir=Path(API.config.FIG_DIR) / model_name_for_dir
    )
    print(f"PDF plots for simulation '{model_name_for_dir}' generated and saved.")


def load_or_train_model(data_base, device, num_epochs, model_name_to_load=None):
    model_class = None
    model_kwargs = {}
    coach_class = None

    if API.config.MODEL_TYPE == 'PILSTM':
        model_class = API.PILSTMNet
        coach_class = API.Coach
        model_kwargs = {
            'input_dim': API.config.NX_LORENZ,
            'output_dim': API.config.N_LORENZ,
            'hidden_size': API.config.LSTM_HIDDEN_SIZE,
            'num_layers': API.config.LSTM_NUM_LAYERS
        }
    elif API.config.MODEL_TYPE == 'Transformer':
        model_class = API.TransformerNet
        coach_class = API.TFCoach
        if API.config.TRANSFORMER_CONFIG == 'SMALL':
            model_kwargs = {
                'input_dim': API.config.NX_LORENZ,
                'output_dim': API.config.N_LORENZ,
                'd_model': API.config.TRANSFORMER_MODEL_DIM_SMALL,
                'num_heads': API.config.TRANSFORMER_NUM_HEADS_SMALL,
                'num_encoder_layers': API.config.TRANSFORMER_NUM_ENCODER_LAYERS_SMALL,
                'num_decoder_layers': API.config.TRANSFORMER_NUM_DECODER_LAYERS_SMALL,
                'd_ff': API.config.TRANSFORMER_FF_DIM_SMALL,
                'dropout_rate': API.config.TRANSFORMER_DROPOUT_RATE
            }
        elif API.config.TRANSFORMER_CONFIG == 'COMPLEX':
            model_kwargs = {
                'input_dim': API.config.NX_LORENZ,
                'output_dim': API.config.N_LORENZ,
                'd_model': API.config.TRANSFORMER_MODEL_DIM_COMPLEX,
                'num_heads': API.config.TRANSFORMER_NUM_HEADS_COMPLEX,
                'num_encoder_layers': API.config.TRANSFORMER_NUM_ENCODER_LAYERS_COMPLEX,
                'num_decoder_layers': API.config.TRANSFORMER_NUM_DECODER_LAYERS_COMPLEX,
                'd_ff': API.config.TRANSFORMER_FF_DIM_COMPLEX,
                'dropout_rate': API.config.TRANSFORMER_DROPOUT_RATE
            }
        else:
            raise ValueError(f"Unknown Transformer configuration: {API.config.TRANSFORMER_CONFIG}")
    else:
        raise ValueError(f"Unknown model type: {API.config.MODEL_TYPE}")

    trained_epochs = 0
    model = None
    optimizer = None
    scaler = None
    scheduler = None
    
    if model_name_to_load:
        model_path = Path(API.config.MODEL_SAVE_DIR) / f"{model_name_to_load}.pth"
        if model_path.exists():
            print(f"Loading model from {model_path} for continued training...")
            model, optimizer, scaler, trained_epochs, scheduler = coach_class.load_model(model_class, model_path, device)
            data_base.scaler = scaler
            print(f"Model loaded. Previously trained for {trained_epochs} epochs.")
        else:
            print(f"Model '{model_name_to_load}' not found at {model_path}. Training a new model.")
    
    if model is None:
        print(f"Starting training of a new {API.config.MODEL_TYPE} model...")
        model = model_class(**model_kwargs)
        coach = coach_class(
            dataBase = data_base, 
            device   = device,
            model    = model,
            total_target_epochs = num_epochs
        )
        coach.train(num_epochs=num_epochs, start_epoch=0)
        coach.save_model(num_epochs)
        model = coach.model
        scaler = data_base.scaler
        trained_epochs = num_epochs
    else:
        print(f"Continuing training for an additional {num_epochs} epochs...")
        new_total_epochs = trained_epochs + num_epochs
        coach = coach_class(
            dataBase = data_base, 
            device   = device,
            model    = model,
            lr       = optimizer.param_groups[0]['lr'],
            total_target_epochs = new_total_epochs
        )
        coach.optimizer.load_state_dict(optimizer.state_dict())
        coach.scheduler.load_state_dict(scheduler.state_dict())
        
        coach.train(num_epochs=new_total_epochs, start_epoch=trained_epochs)
        coach.save_model(new_total_epochs)
        model = coach.model
        scaler = data_base.scaler
        trained_epochs = new_total_epochs

    return model, scaler, trained_epochs
