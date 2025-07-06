import meteo_api as API
import matplotlib.pyplot as plt # type: ignore
import math
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore
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

    def __init__(self, model, dataBase, device, lr=1e-3):
        self.device = device
        self.dataBase = dataBase
        self.model = model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.loss_fn = API.HybridLoss().to(self.device)
        # Set scaler parameters for HybridLoss after dataBase is initialized
        self.loss_fn.set_scaler_params(self.dataBase.scaler.mean_, self.dataBase.scaler.scale_, self.device)
        
        log_dir = Path(API.config.LOGS_DIR) / "tensorboard" / self.get_name()
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

    def get_name(self):
        # Generate a unique name for TensorBoard logs based on model and hyperparameters
        return (
            f"LSTM_N{API.config.N_LORENZ}_Nx{API.config.NX_LORENZ}_F{API.config.F_LORENZ}"
            f"_dt{API.config.DT_LORENZ}_hsize{API.config.LSTM_HIDDEN_SIZE}_layers{API.config.LSTM_NUM_LAYERS}_alphaPI{API.config.ALPHA_PI}"
        )

    def train(self, num_epochs=100):
        self.model.train()
        for epoch in range(num_epochs):
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
            self.writer.add_scalar('Loss/Validation/PhysicsInformed', avg_val_pi_loss, epoch)

            self.model.train() # Set model back to training mode
            # --- End Validation Loop ---
            
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

    def save_model(self, epoch):
        model_filename = f"{self.get_name()}_epoch_{epoch}.pth"
        scaler_filename = f"{self.get_name()}_scaler.joblib"
        
        model_path = Path(API.config.MODEL_SAVE_DIR) / model_filename
        scaler_path = Path(API.config.MODEL_SAVE_DIR) / scaler_filename
        
        model_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'loss_fn_state_dict': self.loss_fn.state_dict(), # If loss_fn has learnable parameters
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
        
        # Load scaler
        # The scaler filename is derived from the model path
        scaler_name_parts = Path(path).stem.split('_epoch')
        base_name = scaler_name_parts[0] if len(scaler_name_parts) > 1 else Path(path).stem
        scaler_path = Path(path).parent / f"{base_name}_scaler.joblib"
        
        scaler = joblib.load(scaler_path)

        print(f"Model and scaler loaded from {path}")
        return model, optimizer, scaler


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
        # predictions_seq: (batch_size, SEGMENT_LENGTH, N_LORENZ) - network's predictions for y(t+1)
        # y_segment_targets: (batch_size, SEGMENT_LENGTH, N_LORENZ) - ground truth for y(t+1)
        # y_segment_current_full_states: (batch_size, SEGMENT_LENGTH, N_LORENZ) - ground truth for y(t)

        if self.scaler_mean is None or self.scaler_scale is None:
            # This should only happen on the very first call if set_scaler_params wasn't called
            # Or if the scaler was loaded after HybridLoss was initialized.
            # For robustness, we can set it here if not already set.
            self.set_scaler_params(scaler.mean_, scaler.scale_, predictions_seq.device)

        # Data-driven loss (on normalized data)
        l_dd = self.mse_loss(
            predictions_seq[:, :, OBS_IDXS],
            y_segment_targets[:, :, OBS_IDXS]
        )

        # Physics-informed loss (on denormalized data)
        # Denormalize predictions_seq (y(t+1)_hat) and y_segment_current_full_states (y(t)_true)
        predictions_seq_denorm = predictions_seq * self.scaler_scale + self.scaler_mean
        y_segment_current_full_states_denorm = y_segment_current_full_states * self.scaler_scale + self.scaler_mean

        # Calculate approximate time derivative (dy/dt) from denormalized predictions
        dydt_approx_denorm = (predictions_seq_denorm - y_segment_current_full_states_denorm) / self.dt_lorenz

        # Calculate f(y) using the denormalized predictions for y(t+1)
        f_y_hat_denorm = self.lorenz96_torch(predictions_seq_denorm, API.config.F_LORENZ)

        # Normalize the physics residual before computing MSE
        dydt_approx_norm = dydt_approx_denorm / self.scaler_scale
        f_y_hat_norm = f_y_hat_denorm / self.scaler_scale

        # Physics-informed loss on normalized residuals
        l_pi = self.mse_loss(dydt_approx_norm, f_y_hat_norm)

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
