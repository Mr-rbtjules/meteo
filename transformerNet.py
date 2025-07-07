import meteo_api as API
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np
import joblib
from meteo_api.dataBase import OBS_IDXS
from torch.utils.tensorboard import SummaryWriter
import math

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
        # x shape: (sequence_length, batch_size, d_model)
        # pe shape: (max_len, 1, d_model)
        # Add positional encoding to the input embeddings
        x = x + self.pe[:x.size(0), :]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask=None):
        # query, key, value shape: (batch_size, sequence_length, d_model)
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => num_heads x d_k
        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # Ensure mask is broadcastable: (batch_size, num_heads, query_len, key_len)
            # If mask is (T, T) (from generate_square_subsequent_mask), unsqueeze to (1, 1, T, T)
            if mask.dim() == 2: # Case for (T, T) mask
                mask = mask.unsqueeze(0).unsqueeze(0) # -> (1, 1, T, T)
            elif mask.dim() == 3: # Case for (B, T, T) mask
                mask = mask.unsqueeze(1) # -> (B, 1, T, T)
            # Expand mask to match scores shape for broadcasting
            mask = mask.expand(batch_size, self.num_heads, scores.size(-2), scores.size(-1))
            scores = scores.masked_fill(mask, float('-inf')) # Use boolean mask directly

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        
        # 3) "Concat" using a view and apply a final linear.
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.dropout(self.linears[-1](x)) # Apply dropout after final linear

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = FeedForward(d_model, d_ff, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        # x shape: (batch_size, sequence_length, d_model)
        # mask shape: (batch_size, 1, sequence_length, sequence_length) or (batch_size, sequence_length, sequence_length)
        
        # Self-attention
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output)) # Add & Norm

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output)) # Add & Norm
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = FeedForward(d_model, d_ff, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x, memory, src_mask, tgt_mask):
        # x shape: (batch_size, target_sequence_length, d_model)
        # memory shape: (batch_size, source_sequence_length, d_model) (encoder output)
        
        # Masked self-attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Cross-attention (query from decoder, key/value from encoder output)
        cross_attn_output = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout_rate):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout_rate)
                                     for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        # x shape: (batch_size, sequence_length, d_model)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout_rate):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout_rate)
                                     for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        # x shape: (batch_size, target_sequence_length, d_model)
        # memory shape: (batch_size, source_sequence_length, d_model)
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class TransformerNet(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff, dropout_rate):
        super(TransformerNet, self).__init__()
        self.model_type = 'Transformer'
        self.input_dim = input_dim # NX_LORENZ
        self.output_dim = output_dim # N_LORENZ
        self.d_model = d_model
        
        self.src_embedding = nn.Linear(input_dim, d_model)
        self.tgt_embedding = nn.Linear(output_dim, d_model) # For decoder input (shifted target)
        self.positional_encoding = PositionalEncoding(d_model)

        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout_rate)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout_rate)
        
        self.output_linear = nn.Linear(d_model, output_dim)

    def generate_square_subsequent_mask(self, sz, device):
        # Returns a boolean mask: True for masked (should be -inf), False for unmasked (should be 0.0)
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
        return mask.to(device)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src shape: (batch_size, src_sequence_length, input_dim)
        # tgt shape: (batch_size, tgt_sequence_length, output_dim) (shifted target for decoder)

        # Embed and add positional encoding
        src = self.positional_encoding(self.src_embedding(src).transpose(0, 1)).transpose(0, 1) # (B, S, D) -> (S, B, D) -> PE -> (S, B, D) -> (B, S, D)
        tgt = self.positional_encoding(self.tgt_embedding(tgt).transpose(0, 1)).transpose(0, 1) # (B, S, D) -> (S, B, D) -> PE -> (S, B, D) -> (B, S, D)

        # Generate masks if not provided
        if src_mask is None:
            # No encoder masking needed
            src_mask = None
        if tgt_mask is None:
            # Decoder causal mask: True for future positions to be masked
            tgt_mask = self.generate_square_subsequent_mask(tgt.shape[1], tgt.device)

        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        predictions = self.output_linear(decoder_output)
        return predictions

class TransformerTrajectoryDataset(Dataset):
    def __init__(self, full_trajectory, raw_trajectory, indices):
        super(TransformerTrajectoryDataset, self).__init__()
        self.full_trajectory = full_trajectory # Normalized data
        self.raw_trajectory = raw_trajectory # Unnormalized data (for physics loss)
        self.indices = indices # Indices of starting points for segments

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        segment_start_idx = self.indices[idx]

        # Input sequence for encoder (observed variables)
        # x(t_0), x(t_1), ..., x(t_{L-1})
        # Shape: (SEGMENT_LENGTH, NX_LORENZ)
        encoder_input = self.full_trajectory[segment_start_idx : segment_start_idx + API.config.SEGMENT_LENGTH, OBS_IDXS]

        # Target sequence for decoder (full state, shifted by one timestep)
        # y(t_1), y(t_2), ..., y(t_L)
        # This is what the decoder should predict
        # Shape: (SEGMENT_LENGTH, N_LORENZ)
        decoder_target = self.full_trajectory[segment_start_idx + 1 : segment_start_idx + API.config.SEGMENT_LENGTH + 1, :]

        # Decoder input (shifted target, observed variables only for the first element, then full state for subsequent)
        # For training, we use teacher forcing. The decoder input is the *true* previous output.
        # The first element of decoder_input is x(t_0) (observed).
        # Subsequent elements are y(t_1), y(t_2), ..., y(t_{L-1}) (full state).
        # This is a common way to handle sequence-to-sequence for forecasting.
        # The decoder input for predicting y(t_k) is y(t_{k-1}).
        # So, for predicting y(t_1)...y(t_L), the input should be y(t_0)...y(t_{L-1}).
        # We use the full state for decoder input, as the decoder predicts the full state.
        # The first element of decoder input is the observed part of y(t_0).
        # The rest are the full states y(t_0) ... y(t_{L-1})
        
        # For the Transformer, the decoder input is typically the target sequence shifted right,
        # with the first element being a start-of-sequence token or the initial observed state.
        # Here, we'll use the observed part of the current full state as the "start" for the decoder input,
        # and then the full state for the rest of the sequence.
        # This means the decoder input for predicting y(t_i+1) is y(t_i).
        # So, for a segment of length L, we need y(t_0) to y(t_{L-1}) as decoder input.
        # The decoder will predict y(t_1) to y(t_L).
        
        # Decoder input: y(t_0), y(t_1), ..., y(t_{L-1})
        # Shape: (SEGMENT_LENGTH, N_LORENZ)
        decoder_input = self.full_trajectory[segment_start_idx : segment_start_idx + API.config.SEGMENT_LENGTH, :]

        # Current full states for physics loss (y(t_0) to y(t_{L-1}))
        # Shape: (SEGMENT_LENGTH, N_LORENZ)
        current_full_states_for_physics = self.full_trajectory[segment_start_idx : segment_start_idx + API.config.SEGMENT_LENGTH, :]

        return (
            torch.tensor(encoder_input, dtype=torch.float32),
            torch.tensor(decoder_input, dtype=torch.float32),
            torch.tensor(decoder_target, dtype=torch.float32),
            torch.tensor(current_full_states_for_physics, dtype=torch.float32)
        )

class TFCoach:
    def __init__(self, model, dataBase, device, lr=1e-3, total_target_epochs=None):
        self.device = device
        self.dataBase = dataBase
        self.model = model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=API.config.LR_SCHEDULER_FACTOR,
            patience=API.config.LR_SCHEDULER_PATIENCE,
            verbose=True
        )

        self.loss_fn = API.HybridLoss().to(self.device)
        self.loss_fn.set_scaler_params(self.dataBase.scaler.mean_, self.dataBase.scaler.scale_, self.device)
        
        self.writer = None
        if total_target_epochs is not None:
            log_dir = Path(API.config.LOGS_DIR) / "tensorboard" / self.get_name(total_target_epochs)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))

        if self.model.model_type == 'Transformer':
            self.train_dataset = TransformerTrajectoryDataset(self.dataBase.trajectory, self.dataBase.raw_trajectory, self.dataBase.train_lstm_segment_indices)
            self.val_dataset = TransformerTrajectoryDataset(self.dataBase.trajectory, self.dataBase.raw_trajectory, self.dataBase.val_lstm_segment_indices)
        else:
            raise ValueError(f"Unknown model type: {self.model.model_type}")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=API.config.BATCH_SIZE,
            shuffle=True,
            num_workers=API.config.NUM_WORKERS,
            worker_init_fn=API.dataBase.worker_init_fn
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=API.config.BATCH_SIZE,
            shuffle=False,
            num_workers=API.config.NUM_WORKERS,
            worker_init_fn=API.dataBase.worker_init_fn
        )

    def get_name(self, num_epochs):
        return (
            f"Transformer_alphaPI{API.config.ALPHA_PI}"
            f"_seqLen{API.config.SEGMENT_LENGTH}"
            f"_epochs{num_epochs}"
        )

    def train(self, num_epochs=100, start_epoch=0):
        self.model.train()
        for epoch in range(start_epoch, num_epochs):
            total_loss_epoch = 0
            total_dd_loss_epoch = 0
            total_pi_loss_epoch = 0
            
            for batch_idx, (encoder_input, decoder_input, decoder_target, current_full_states_for_physics) in enumerate(self.train_loader):
                encoder_input = encoder_input.to(self.device)
                decoder_input = decoder_input.to(self.device)
                decoder_target = decoder_target.to(self.device)
                current_full_states_for_physics = current_full_states_for_physics.to(self.device)

                self.optimizer.zero_grad()

                predictions_seq = self.model(encoder_input, decoder_input)

                total_loss, l_dd, l_pi = self.loss_fn(
                    predictions_seq, 
                    decoder_target, 
                    current_full_states_for_physics,
                    self.dataBase.scaler
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
            
            self.model.eval()
            val_total_loss_epoch = 0
            val_dd_loss_epoch = 0
            val_pi_loss_epoch = 0
            with torch.no_grad():
                for batch_idx_val, (encoder_input_val, decoder_input_val, decoder_target_val, current_full_states_for_physics_val) in enumerate(self.val_loader):
                    encoder_input_val = encoder_input_val.to(self.device)
                    decoder_input_val = decoder_input_val.to(self.device)
                    decoder_target_val = decoder_target_val.to(self.device)
                    current_full_states_for_physics_val = current_full_states_for_physics_val.to(self.device)

                    predictions_seq_val = self.model(encoder_input_val, decoder_input_val)

                    val_total_loss, val_l_dd, val_l_pi = self.loss_fn(
                        predictions_seq_val, 
                        decoder_target_val, 
                        current_full_states_for_physics_val, 
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
            self.scheduler.step(avg_val_total_loss)

            self.model.train()
            
        if self.writer:
            self.writer.close()
        print("Training complete.")
        self.save_model(num_epochs)

    def predict_closed_loop(self, initial_state, num_simulation_steps):
        self.model.eval()
        with torch.no_grad():
            initial_state_normalized = self.dataBase.scaler.transform(initial_state.reshape(1, -1)).flatten()

            # Encoder seed uses only observed components
            current_encoder_input = torch.tensor(
                initial_state_normalized[OBS_IDXS], dtype=torch.float32, device=self.device
            ).unsqueeze(0).unsqueeze(0)
            # Decoder seed: zeros for unobserved, observed for OBS_IDXS
            decoder0 = torch.zeros((1, 1, self.model.output_dim), dtype=torch.float32, device=self.device)
            decoder0[..., OBS_IDXS] = torch.tensor(initial_state_normalized[OBS_IDXS], dtype=torch.float32, device=self.device)
            current_decoder_input = decoder0

            # No masking on encoder at inference
            src_mask = None
            # Causal mask for one-step decoder
            tgt_mask = self.model.generate_square_subsequent_mask(1, self.device)

            predicted_next_state_normalized = self.model(
                current_encoder_input,
                current_decoder_input,
                src_mask=src_mask,
                tgt_mask=tgt_mask
            )
            predicted_next_state_normalized = predicted_next_state_normalized.squeeze(0).squeeze(0)

            simulated_trajectory_tensors = [torch.tensor(initial_state_normalized, dtype=torch.float32, device=self.device)]

            for step in range(num_simulation_steps):
                # No masking on encoder at inference
                src_mask = None
                tgt_mask = self.model.generate_square_subsequent_mask(1, self.device)

                predicted_next_state_normalized = self.model(
                    current_encoder_input,
                    current_decoder_input,
                    src_mask=src_mask,
                    tgt_mask=tgt_mask
                )
                predicted_next_state_normalized = predicted_next_state_normalized.squeeze(0).squeeze(0)

                current_encoder_input = predicted_next_state_normalized[OBS_IDXS].unsqueeze(0).unsqueeze(0)
                current_decoder_input = predicted_next_state_normalized.unsqueeze(0).unsqueeze(0)

                simulated_trajectory_tensors.append(predicted_next_state_normalized)

            simulated_trajectory_normalized_tensor = torch.stack(simulated_trajectory_tensors)
            simulated_trajectory_normalized_numpy = simulated_trajectory_normalized_tensor.cpu().numpy()
            simulated_trajectory_original_scale = self.dataBase.scaler.inverse_transform(simulated_trajectory_normalized_numpy)

            return simulated_trajectory_original_scale

    def save_model(self, total_epochs_trained):
        model_filename = f"{self.get_name(total_epochs_trained)}.pth"
        scaler_filename = f"{self.get_name(total_epochs_trained)}_scaler.joblib"
        
        model_path = Path(API.config.MODEL_SAVE_DIR) / model_filename
        scaler_path = Path(API.config.MODEL_SAVE_DIR) / scaler_filename
        
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'epoch': total_epochs_trained,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, model_path)
        
        joblib.dump(self.dataBase.scaler, scaler_path)
        print(f"Model and scaler saved to {model_path} and {scaler_path}")

    @staticmethod
    def load_model(model_class, path, device):
        checkpoint = torch.load(path, map_location=device)
        
        model_kwargs = {
            'input_dim': API.config.NX_LORENZ,
            'output_dim': API.config.N_LORENZ,
            'd_model': API.config.TRANSFORMER_MODEL_DIM,
            'num_heads': API.config.TRANSFORMER_NUM_HEADS,
            'num_encoder_layers': API.config.TRANSFORMER_NUM_ENCODER_LAYERS,
            'num_decoder_layers': API.config.TRANSFORMER_NUM_DECODER_LAYERS,
            'd_ff': API.config.TRANSFORMER_FF_DIM,
            'dropout_rate': API.config.TRANSFORMER_DROPOUT_RATE
        }

        model = model_class(**model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        optimizer = optim.Adam(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=API.config.LR_SCHEDULER_FACTOR,
            patience=API.config.LR_SCHEDULER_PATIENCE,
            verbose=True
        )
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        base_name = Path(path).stem
        scaler_path = Path(path).parent / f"{base_name}_scaler.joblib"
        
        scaler = joblib.load(scaler_path)

        print(f"Model and scaler loaded from {path}")
        return model, optimizer, scaler, checkpoint['epoch'], scheduler
