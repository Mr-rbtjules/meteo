
import meteo_api as API
import matplotlib.pyplot as plt # type: ignore
import math
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore
from pathlib import Path
from torch.utils.data import DataLoader # Added import
import numpy as np # type: ignore

from torch.utils.tensorboard import SummaryWriter # type: ignore


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (sequence_length, batch_size, embedding_dim)
        return x + self.pe[:x.size(0), :]


class HybridNet(nn.Module):
    """
    A Physics‑Informed Transformer model for autoregressive forecasting.

    Default mode is an encoder‑only (GPT‑style) architecture with causal
    masking.  Set ``use_kv_cache=True`` to enable an incremental KV‑cache
    for faster closed‑loop inference.
    """
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers,
                 dim_feedforward=2048, dropout=0.1, *, use_kv_cache: bool = False):
        super(HybridNet, self).__init__()
        self.use_kv_cache = use_kv_cache
        self.kv_cache = None  # will hold past embeddings when KV‑cache is enabled
        self.model_type = 'DecoderOnlyTransformer'
        self.input_dim = input_dim # NX_LORENZ
        self.output_dim = output_dim # N_LORENZ
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Causal Transformer Encoder (decoder‑only style)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_linear = nn.Linear(d_model, output_dim)

    def forward(self, src, *, reset_cache: bool = False):
        """
        Args
        ----
        src : Tensor
            Shape (batch_size, sequence_length, input_dim).
        reset_cache : bool, optional
            If True, forces the internal KV‑cache to be cleared
            at the beginning of a new autoregressive rollout.
        """
        if reset_cache or not self.use_kv_cache or self.training:
            self.kv_cache = None  # (re)start a fresh cache when training or if disabled

        # --- (Optional) build / extend KV cache -------------------------------------
        if self.use_kv_cache and not self.training:
            # keep only the last token to append
            last_token = src[:, -1:, :]  # (B, 1, input_dim)

            if self.kv_cache is None:
                working_src = last_token
            else:
                working_src = torch.cat([self.kv_cache, last_token], dim=1)

            # store for next call
            self.kv_cache = working_src
        else:
            working_src = src  # full context (training or cache disabled)

        # ---------------------------------------------------------------------------
        # Transformer expects (seq_len, batch, feat) when batch_first=False
        working_src = working_src.permute(1, 0, 2)

        # input projection + pos‑enc
        emb = self.input_embedding(working_src) * math.sqrt(self.d_model)
        emb = self.pos_encoder(emb)

        # causal mask
        seq_len = emb.size(0)
        src_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=emb.device)

        # encoder pass
        enc_out = self.transformer_encoder(emb, mask=src_mask)

        # take the representation corresponding to the latest timestep
        last_step_repr = enc_out[-1, :, :]          # (batch, d_model)
        prediction = self.output_linear(last_step_repr)  # (batch, output_dim)
        return prediction


class Coach:

    def __init__(self, dataBase, device, lr=1e-3):
        self.device = device
        self.dataBase = dataBase
        
        # Initialize the Transformer model
        self.model = HybridNet(
            input_dim=API.config.NX_LORENZ,
            output_dim=API.config.N_LORENZ,
            d_model=API.config.EMBEDDING_DIM,
            nhead=API.config.NUM_HEADS,
            num_layers=API.config.NUM_LAYERS
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.loss_fn = API.HybridLoss().to(self.device)
        
        log_dir = Path(API.config.LOGS_DIR) / "tensorboard" / self.get_name()
        log_dir.mkdir(parents=True, exist_ok=True)  # create directory if it doesn't exist
        self.writer = SummaryWriter(log_dir=str(log_dir))

        self.train_dataset = API.DataBase.TrajectoryDataset(self.dataBase.reduced)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=API.config.BATCH_SIZE,
            shuffle=True,
            num_workers=API.config.NUM_WORKERS,
            worker_init_fn=API.dataBase.worker_init_fn # Ensure reproducibility
        )

    def get_name(self):
        # Generate a unique name for TensorBoard logs based on model and hyperparameters
        return (
            f"Transformer_N{API.config.N_LORENZ}_Nx{API.config.NX_LORENZ}_F{API.config.F_LORENZ}"
            f"_dt{API.config.DT_LORENZ}_seq{API.config.SEQUENCE_LENGTH}"
            f"_dmodel{API.config.EMBEDDING_DIM}_nhead{API.config.NUM_HEADS}"
            f"_layers{API.config.NUM_LAYERS}_alphaPI{API.config.ALPHA_PI}"
        )

    def train(self, num_epochs=100):
        self.model.train()
        for epoch in range(num_epochs):
            total_loss_epoch = 0
            total_dd_loss_epoch = 0
            total_pi_loss_epoch = 0
            
            for batch_idx, (segment_x_seqs, segment_y_nexts, segment_y_current_full_states) in enumerate(self.train_loader):
                # segment_x_seqs: (batch_size, SEGMENT_LENGTH, SEQUENCE_LENGTH, NX_LORENZ)
                # segment_y_nexts: (batch_size, SEGMENT_LENGTH, N_LORENZ)
                # segment_y_current_full_states: (batch_size, SEGMENT_LENGTH, N_LORENZ)

                # Reshape to (batch_size * SEGMENT_LENGTH, SEQUENCE_LENGTH, NX_LORENZ)
                # and (batch_size * SEGMENT_LENGTH, N_LORENZ) for targets
                batch_size, segment_length, seq_len, nx = segment_x_seqs.shape
                _, _, n_lorenz = segment_y_nexts.shape

                x_seq_flat = segment_x_seqs.view(-1, seq_len, nx).to(self.device)
                y_next_flat = segment_y_nexts.view(-1, n_lorenz).to(self.device)
                y_current_full_state_flat = segment_y_current_full_states.view(-1, n_lorenz).to(self.device)

                self.optimizer.zero_grad()
                
                # Forward pass for all sequences in the flattened batch
                predictions_flat = self.model(x_seq_flat) # predictions_flat is ŷ(t+1) for each step in the segment

                # Calculate loss for the entire flattened batch
                total_loss, l_dd, l_pi = self.loss_fn(predictions_flat, y_next_flat, y_current_full_state_flat)
                
                # Backward pass and optimize
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
            
        self.writer.close()
        print("Training complete.")

    def predict_closed_loop(self, initial_state, num_simulation_steps):
        self.model.eval() # Set model to evaluation mode
        with torch.no_grad():
            # initial_state: numpy array (N_LORENZ,)
            # We need to normalize it first
            initial_state_normalized = self.dataBase.scaler.transform(initial_state.reshape(1, -1)).flatten()
            
            # The first input to the Transformer is a sequence of observed variables.
            # For closed-loop, we start with the initial state's observed variables,
            # and for the preceding SEQUENCE_LENGTH-1 steps, we can use the initial state's observed variables
            # or zeros, depending on how we want to "prime" the model.
            # A simple approach is to repeat the initial observed state for the sequence length.
            # Or, if we have a true initial sequence, we would use that.
            # For now, let's assume we are given a single initial state y(t_0) and want to predict y(t_1), y(t_2), ...
            # The model expects a sequence of length SEQUENCE_LENGTH.
            # So, we'll create an initial input sequence by repeating the observed part of the initial state.
            
            # This is a simplification. A more robust approach would be to provide a true initial sequence
            # if available, or to use a warm-up period.
            # For now, let's use the initial_state's observed part as the last element of the input sequence,
            # and fill the rest with zeros or a sensible default.
            # The paper says: "The Transformer predicts the next full state ỹ(t_1) from x(t_0).
            # Then, it uses ỹ(t_1) as the input to predict ỹ(t_2), and so on."
            # This implies the input is always just x(t).
            # So, the input to the model is (batch_size, sequence_length, NX_LORENZ).
            # For the first prediction, we need a sequence of x(t) up to t_0.
            # Let's assume initial_state is y(t_0). We need x(t_0) as the last element of the input sequence.
            # For simplicity, let's create an input sequence where all elements are x(t_0).
            
            # The model's forward method expects (batch_size, sequence_length, input_dim)
            # The input_dim is NX_LORENZ.
            
            # Initialize the trajectory with the initial state
            simulated_trajectory = [initial_state_normalized]
            
            # Create the initial input sequence for the Transformer
            # This sequence will be updated iteratively
            current_input_sequence = torch.zeros(API.config.SEQUENCE_LENGTH, API.config.NX_LORENZ, dtype=torch.float32).to(self.device)
            
            # Set the last element of the input sequence to the observed part of the initial state
            current_input_sequence[-1, :] = torch.tensor(initial_state_normalized[:API.config.NX_LORENZ], dtype=torch.float32).to(self.device)

            for step in range(num_simulation_steps):
                # Add batch dimension
                input_batch = current_input_sequence.unsqueeze(0) # Shape: (1, SEQUENCE_LENGTH, NX_LORENZ)
                
                # Predict the next full state
                predicted_next_state_normalized = self.model(input_batch).squeeze(0) # Shape: (N_LORENZ,)
                
                # Add the predicted state to the simulated trajectory
                simulated_trajectory.append(predicted_next_state_normalized.cpu().numpy())
                
                # Update the input sequence for the next prediction
                # Shift the sequence and add the observed part of the new prediction
                current_input_sequence = torch.roll(current_input_sequence, shifts=-1, dims=0)
                current_input_sequence[-1, :] = predicted_next_state_normalized[:API.config.NX_LORENZ]
                
            # Convert list of numpy arrays to a single numpy array
            simulated_trajectory_normalized = np.array(simulated_trajectory)
            
            # Inverse transform the normalized trajectory to get original scale
            simulated_trajectory_original_scale = self.dataBase.scaler.inverse_transform(simulated_trajectory_normalized)
            
            return simulated_trajectory_original_scale


class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.n_lorenz = API.config.N_LORENZ
        self.nx_lorenz = API.config.NX_LORENZ
        self.dt_lorenz = API.config.DT_LORENZ
        self.alpha_pi = API.config.ALPHA_PI
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets, y_current_full_state):
        # predictions: ŷ(t+1) from the model (batch_size, N_LORENZ)
        # targets: y(t+1) ground truth (batch_size, N_LORENZ)
        # y_current_full_state: y(t_i) from the dataset (batch_size, N_LORENZ)

        # Data-driven loss L_dd: MSE on observed variables only
        # We only penalize the first NX_LORENZ variables
        l_dd = self.mse_loss(predictions[:, :self.nx_lorenz], targets[:, :self.nx_lorenz])

        # Physics-informed loss L_pi
        # Approximate dŷ/dt via finite difference: (ỹ(t+1) - ỹ(t_i)) / Δt
        # Note: predictions is ỹ(t+1), y_current_full_state is ỹ(t_i)
        dydt_approx = (predictions - y_current_full_state) / self.dt_lorenz

        # Calculate f(ŷ) using Lorenz-96 ODE for each item in the batch
        # We need to apply the ODE function to each row of 'predictions'
        # Since lorenz_96_ode is a numpy function, we need to convert to numpy and back to tensor.
        # This might be slow, but for a plan, it outlines the logic.
        f_y_hat = torch.zeros_like(predictions)
        for i in range(predictions.shape[0]):
            # Detach predictions to avoid computing gradients through the ODE function itself
            # The physics loss penalizes the *output* of the network for not matching the ODE,
            # not the ODE itself.
            f_y_hat[i] = torch.tensor(API.tools.lorenz_96_ode(predictions[i].detach().cpu().numpy(), API.config.F_LORENZ), dtype=torch.float32).to(predictions.device)

        l_pi = self.mse_loss(dydt_approx, f_y_hat)

        # Total loss
        total_loss = l_dd + self.alpha_pi * l_pi
        return total_loss, l_dd, l_pi
