import meteo_api as API
import matplotlib.pyplot as plt # type: ignore
import math
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore
from torch.optim import lr_scheduler # Added import for learning rate scheduler
from pathlib import Path
from torch.utils.data import Dataset, DataLoader # Added import
import numpy as np # type: ignore
import joblib # For saving/loading scaler
from meteo_api.dataBase import OBS_IDXS
from torch.utils.tensorboard import SummaryWriter # type: ignore


class PILSTMNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, num_layers, dropout=API.config.LSTM_DROPOUT_RATE):
        super(PILSTMNet, self).__init__()
        self.model_type = 'PILSTM'
        self.input_dim = input_dim # NX_LORENZ
        self.output_dim = output_dim # N_LORENZ
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.output_linear = nn.Linear(hidden_size, output_dim)

    def forward(self, src, hidden=None):
        # src shape: (batch_size, sequence_length, input_dim)
        
        lstm_out, hidden = self.lstm(src, hidden)
        
        # Take the output from the last timestep of the LSTM
        # For training, we need outputs for all timesteps in the sequence
        # For inference, we only need the last one
        # So, return full sequence output for training, and handle last step in Coach
        
        # If src is a sequence (batch_size, seq_len, input_dim), lstm_out is (batch_size, seq_len, hidden_size)
        # We need to apply linear layer to each timestep's output
        predictions_seq = self.output_linear(lstm_out) # Shape: (batch_size, sequence_length, N_LORENZ)
        
        return predictions_seq, hidden


class Coach:

    def __init__(self, model, dataBase, device, lr=1e-3, total_target_epochs=None):
        self.device = device
        self.dataBase = dataBase
        self.model = model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Initialize learning rate scheduler
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min', # Monitor validation loss
            factor=API.config.LR_SCHEDULER_FACTOR,
            patience=API.config.LR_SCHEDULER_PATIENCE,
            verbose=True
        )

        self.loss_fn = API.HybridLoss().to(self.device)
        # Set scaler parameters for HybridLoss after dataBase is initialized
        self.loss_fn.set_scaler_params(self.dataBase.scaler.mean_, self.dataBase.scaler.scale_, self.device)
        
        # Only initialize SummaryWriter if total_target_epochs is provided (i.e., for training runs)
        self.writer = None # Initialize to None
        if total_target_epochs is not None:
            log_dir = Path(API.config.LOGS_DIR) / "tensorboard" / self.get_name(total_target_epochs)
            log_dir.mkdir(parents=True, exist_ok=True)  # create directory if it doesn't exist
            self.writer = SummaryWriter(log_dir=str(log_dir))

        # Conditional dataset and dataloader initialization
        if self.model.model_type == 'PILSTM':
            self.train_dataset = API.LSTMTrajectoryDataset(self.dataBase.trajectory, self.dataBase.raw_trajectory, self.dataBase.train_lstm_segment_indices)
            self.val_dataset = API.LSTMTrajectoryDataset(self.dataBase.trajectory, self.dataBase.raw_trajectory, self.dataBase.val_lstm_segment_indices)
        else:
            raise ValueError(f"Unknown model type: {self.model.model_type}")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=API.config.BATCH_SIZE,
            shuffle=True,
            num_workers=API.config.NUM_WORKERS,
            worker_init_fn=API.dataBase.worker_init_fn # Ensure reproducibility
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=API.config.BATCH_SIZE,
            shuffle=False, # No need to shuffle validation data
            num_workers=API.config.NUM_WORKERS,
            worker_init_fn=API.dataBase.worker_init_fn
        )

    def get_name(self, num_epochs):
        # Generate a unique name for TensorBoard logs and saved models based on relevant hyperparameters
        return (
            f"LSTM_alphaPI{API.config.ALPHA_PI}"
            f"_seqLen{API.config.SEGMENT_LENGTH}"
            f"_epochs{num_epochs}"
        )

    def train(self, num_epochs=100, start_epoch=0):
        self.model.train()
        for epoch in range(start_epoch, num_epochs):
            total_loss_epoch = 0
            total_dd_loss_epoch = 0
            total_pi_loss_epoch = 0
            
            for batch_idx, (x_segment, y_segment_targets, y_segment_current_full_states) in enumerate(self.train_loader):
                # x_segment: (batch_size, SEGMENT_LENGTH, NX_LORENZ)
                # y_segment_targets: (batch_size, SEGMENT_LENGTH, N_LORENZ)
                # y_segment_current_full_states: (batch_size, SEGMENT_LENGTH, N_LORENZ)

                x_segment = x_segment.to(self.device)
                y_segment_targets = y_segment_targets.to(self.device)
                y_segment_current_full_states = y_segment_current_full_states.to(self.device)

                self.optimizer.zero_grad()

                # Unroll LSTM over the segment
                # predictions_seq will be (batch_size, SEGMENT_LENGTH, N_LORENZ)
                predictions_seq, _ = self.model(x_segment) 

                # Calculate loss for the entire sequence
                total_loss, l_dd, l_pi = self.loss_fn(
                    predictions_seq, 
                    y_segment_targets, 
                    y_segment_current_full_states, 
                    self.dataBase.scaler
                )

                total_loss.backward()
                self.optimizer.step()

                total_loss_epoch += total_loss.item()
                total_dd_loss_epoch += l_dd.item()
                total_pi_loss_epoch += l_pi.item()

            avg_total_loss = total_loss_epoch / len(self.train_loader)
            avg_dd_loss = total_dd_loss_epoch / len(self.train_loader)
            avg_pi_loss = total_pi_loss_epoch / len(self.train_loader)

            print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {avg_total_loss:.6f}, "
                  f"Data-driven Loss: {avg_dd_loss:.6f}, Physics-informed Loss: {avg_pi_loss:.6f}")
            
            self.writer.add_scalar('Loss/Total', avg_total_loss, epoch)
            self.writer.add_scalar('Loss/DataDriven', avg_dd_loss, epoch)
            self.writer.add_scalar('Loss/PhysicsInformed', avg_pi_loss, epoch)
            
            # --- Validation Loop ---
            self.model.eval() # Set model to evaluation mode
            val_total_loss_epoch = 0
            val_dd_loss_epoch = 0
            val_pi_loss_epoch = 0
            with torch.no_grad():
                for batch_idx_val, (x_segment_val, y_segment_targets_val, y_segment_current_full_states_val) in enumerate(self.val_loader):
                    x_segment_val = x_segment_val.to(self.device)
                    y_segment_targets_val = y_segment_targets_val.to(self.device)
                    y_segment_current_full_states_val = y_segment_current_full_states_val.to(self.device)

                    predictions_seq_val, _ = self.model(x_segment_val)

                    val_total_loss, val_l_dd, val_l_pi = self.loss_fn(
                        predictions_seq_val, 
                        y_segment_targets_val, 
                        y_segment_current_full_states_val, 
                        self.dataBase.scaler
                    )
                    
                    val_total_loss_epoch += val_total_loss.item()
                    val_dd_loss_epoch += val_l_dd.item()
                    val_pi_loss_epoch += val_l_pi.item()

            avg_val_total_loss = val_total_loss_epoch / len(self.val_loader)
            avg_val_dd_loss = val_dd_loss_epoch / len(self.val_loader)
            avg_val_pi_loss = val_pi_loss_epoch / len(self.val_loader)

            print(f"Epoch {epoch+1}/{num_epochs}, Validation Total Loss: {avg_val_total_loss:.6f}, "
                  f"Val Data-driven Loss: {avg_val_dd_loss:.6f}, Val Physics-informed Loss: {avg_val_pi_loss:.6f}")
            
            self.writer.add_scalar('Loss/Validation/Total', avg_val_total_loss, epoch)
            self.writer.add_scalar('Loss/Validation/DataDriven', avg_val_dd_loss, epoch)
            # Step the learning rate scheduler
            self.scheduler.step(avg_val_total_loss)

            self.model.train() # Set model back to training mode
            # --- End Validation Loop ---
            
        if self.writer: # Only close writer if it was initialized
            self.writer.close()
        print("Training complete.")
        self.save_model(num_epochs) # Save model after final epoch

    def predict_closed_loop(self, initial_state, num_simulation_steps):
        self.model.eval() # Set model to evaluation mode
        with torch.no_grad():
            # initial_state: numpy array (N_LORENZ)
            # We need to normalize it first
            initial_state_normalized = self.dataBase.scaler.transform(initial_state.reshape(1, -1)).flatten()
            
            # Initialize the trajectory with the initial state as a tensor on the device
            initial_state_tensor = torch.tensor(initial_state_normalized, dtype=torch.float32).to(self.device)
            simulated_trajectory_tensors = [initial_state_tensor]
            
            # For LSTM, we need to maintain hidden state
            hidden = None
            # The input sequence for LSTM is just the current observed state
            current_lstm_input = initial_state_tensor[OBS_IDXS].unsqueeze(0).unsqueeze(0)
            #current_lstm_input = initial_state_tensor[:API.config.NX_LORENZ].unsqueeze(0).unsqueeze(0) # (1, 1, NX_LORENZ)
            
            for step in range(num_simulation_steps):
                # For LSTM, input is just the current observed state
                # current_lstm_input shape: (1, 1, NX_LORENZ)
                predicted_next_state_normalized, hidden = self.model(current_lstm_input, hidden) # LSTM returns (output, hidden)
                predicted_next_state_normalized = predicted_next_state_normalized.squeeze(0).squeeze(0) # Shape: (N_LORENZ,)
                
                # The next input for LSTM is the observed part of the current prediction
                current_lstm_input = predicted_next_state_normalized[OBS_IDXS].unsqueeze(0).unsqueeze(0)

                # Add the predicted state to the simulated trajectory (keep as tensor on device)
                simulated_trajectory_tensors.append(predicted_next_state_normalized)
                
            # Convert list of tensors to a single tensor
            simulated_trajectory_normalized_tensor = torch.stack(simulated_trajectory_tensors) # Shape: (num_steps, N_LORENZ)
            
            # Move to CPU and convert to numpy for inverse_transform, only once at the end
            simulated_trajectory_normalized_numpy = simulated_trajectory_normalized_tensor.cpu().numpy()
            
            # Inverse transform the normalized trajectory to get original scale
            simulated_trajectory_original_scale = self.dataBase.scaler.inverse_transform(simulated_trajectory_normalized_numpy)
            
            return simulated_trajectory_original_scale

    def save_model(self, total_epochs_trained):
        model_filename = f"{self.get_name(total_epochs_trained)}.pth"
        scaler_filename = f"{self.get_name(total_epochs_trained)}_scaler.joblib"
        
        model_path = Path(API.config.MODEL_SAVE_DIR) / model_filename
        scaler_path = Path(API.config.MODEL_SAVE_DIR) / scaler_filename
        
        model_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        
        torch.save({
            'epoch': total_epochs_trained,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(), # Save scheduler state
        }, model_path)
        
        joblib.dump(self.dataBase.scaler, scaler_path)
        print(f"Model and scaler saved to {model_path} and {scaler_path}")

    @staticmethod
    def load_model(model_class, path, device):
        checkpoint = torch.load(path, map_location=device)
        
        # Determine model_kwargs based on model_class type
        if model_class == PILSTMNet:
            model_kwargs = {
                'input_dim': API.config.NX_LORENZ,
                'output_dim': API.config.N_LORENZ,
                'hidden_size': API.config.LSTM_HIDDEN_SIZE,
                'num_layers': API.config.LSTM_NUM_LAYERS
            }
        else:
            raise ValueError(f"Unknown model class for loading: {model_class}")

        model = model_class(**model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        optimizer = optim.Adam(model.parameters()) # Re-initialize optimizer with model parameters
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=API.config.LR_SCHEDULER_FACTOR,
            patience=API.config.LR_SCHEDULER_PATIENCE,
            verbose=True
        )
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler
        # The scaler filename is derived from the model path
        # It's now simpler as the epoch is part of the base name
        base_name = Path(path).stem
        scaler_path = Path(path).parent / f"{base_name}_scaler.joblib"
        
        scaler = joblib.load(scaler_path)

        print(f"Model and scaler loaded from {path}")
        return model, optimizer, scaler, checkpoint['epoch'], scheduler # Return scheduler


class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.n_lorenz = API.config.N_LORENZ
        self.nx_lorenz = API.config.NX_LORENZ
        self.dt_lorenz = API.config.DT_LORENZ
        self.alpha_pi = API.config.ALPHA_PI
        self.mse_loss = nn.MSELoss()
        
        # These will be set by the Coach after DataBase is initialized
        self.scaler_mean = None
        self.scaler_scale = None

    def set_scaler_params(self, scaler_mean, scaler_scale, device):
        self.scaler_mean = torch.tensor(scaler_mean, dtype=torch.float32).to(device)
        self.scaler_scale = torch.tensor(scaler_scale, dtype=torch.float32).to(device)

    def forward(self, predictions_seq, y_segment_targets, y_segment_current_full_states, scaler):
        """
        Computes the hybrid loss on physical‑scale (denormalised) quantities:
        - Data-driven loss (MSE on observed variables, physical scale)
        - Physics-informed loss (MSE between finite-difference derivative and Lorenz-96 RHS, both physical scale).
        The y_segment_current_full_states argument is no longer required for the PI term (retained for backwards compatibility).
        """
        if self.scaler_mean is None or self.scaler_scale is None:
            # This should only happen on the very first call if set_scaler_params wasn't called
            # Or if the scaler was loaded after HybridLoss was initialized.
            # For robustness, we can set it here if not already set.
            self.set_scaler_params(scaler.mean_, scaler.scale_, predictions_seq.device)

        # ---- Denormalise predictions & targets to physical scale ----
        predictions_seq_denorm = predictions_seq * self.scaler_scale + self.scaler_mean  # (B, S, N)
        y_segment_targets_denorm = y_segment_targets * self.scaler_scale + self.scaler_mean  # (B, S, N)

        # Data‑driven loss (physical units)
        l_dd = self.mse_loss(
            predictions_seq_denorm[:, :, OBS_IDXS],
            y_segment_targets_denorm[:, :, OBS_IDXS]
        )
                # ---- Physics-informed loss (on physical scale) ----
        # keep: predictions_seq_denorm -> ŷ(t_i+1)
# Step length is S.  We can only form derivatives for i = 0 … S-2.
        y_pred_now  = predictions_seq_denorm[:, :-1, :]   # ŷ(t_i)
        y_pred_next = predictions_seq_denorm[:, 1:,  :]   # ŷ(t_i+1)

        dy_dt_pred  = (y_pred_next - y_pred_now) / API.config.DT_LORENZ  # (B, S-1, N)
        f_y_pred    = self.lorenz96_torch(y_pred_now, API.config.F_LORENZ)

        l_pi = self.mse_loss(dy_dt_pred, f_y_pred)

        # Finite-difference derivative  (ŷ(t_i+1) − y(t_i)) / Δt
        

        """# ---- Physics‑informed loss (on physical scale) ----
        
        # predictions_seq_denorm already available

        # Finite‑difference time derivative *using consecutive predictions only*
        # Δŷ/Δt gives an approximation of ẏ(t)
        dy_dt_approx_denorm = (
            predictions_seq_denorm[:, 1:, :] - predictions_seq_denorm[:, :-1, :]
        ) / self.dt_lorenz  # (B, S‑1, N)

        # Evaluate Lorenz‑96 RHS at predicted states ŷ(t)
        f_y_hat_denorm = self.lorenz96_torch(
            predictions_seq_denorm[:, :-1, :], API.config.F_LORENZ
        )  # (B, S‑1, N)

        
        # Physics-informed loss directly on physical-scale residuals
        l_pi = self.mse_loss(dy_dt_approx_denorm, f_y_hat_denorm)
"""
        # Total loss
        total_loss = l_dd + self.alpha_pi * l_pi
        return total_loss, l_dd, l_pi

    
    def lorenz96_torch(self, y, F):
        """
        Vectorised Lorenz-96 right-hand side.
        y : (batch, N)  — any device / dtype
        F : forcing term

        returns : dy/dt  (same shape)
        """
        # always roll along the last (state-variable) axis
        y_ip1 = torch.roll(y, shifts=-1, dims=-1)  # y_{i+1}
        y_im1 = torch.roll(y, shifts=1,  dims=-1)  # y_{i-1}
        y_im2 = torch.roll(y, shifts=2,  dims=-1)  # y_{i-2}
        return (y_ip1 - y_im2) * y_im1 - y + F
