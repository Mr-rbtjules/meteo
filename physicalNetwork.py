
import meteo_api as API
import matplotlib.pyplot as plt # type: ignore
import math
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore
from torch.utils.tensorboard import SummaryWriter # type: ignore
from pathlib import Path
import torch.profiler

class HybridNet(nn.Module):
    def __init__(
            self, seq_len=1000, input_dim=4, 
            embed_dim=128, num_layers=2, num_heads=4, 
            physnet_hidden=64, num_tokens=1, attention_learn=False
        ):
        super(HybridNet, self).__init__()
        #create a Hypernetwork (transformers)
        self.hypernet = HyperNet(seq_len=seq_len,
                                 input_dim=input_dim,
                                 embed_dim=embed_dim,
                                 num_layers=num_layers,
                                 num_heads=num_heads,
                                 physnet_hidden=physnet_hidden,
                                 num_tokens=num_tokens,
                                 attention_learn=attention_learn)  # pass num_tokens if you decide to use more
        #create a Physical Network (simple perceptron layers)
        self.physnet = PhysNet(physnet_hidden=physnet_hidden)
        print("HybridNet initialized")
    
    def forward(self, context, t):
        # Generate physics network parameters from context using the hypernetwork
        params = self.hypernet(context)
        # Compute the prediction at time t using the dynamic physics network
        pred = self.physnet(t, params)
        return pred


class HyperNet(nn.Module):
    """
    A transformer-based hyper-network that generates the parameters
    (weights and biases) for the physics network.
    
    The input to the hyper-net is a batch of spatiotemporal context points (e.g., 1000 points,
    each with [x, y, z, t]). The network first embeds the inputs (adding a fixed positional encoding),
    prepends a learnable token, and processes the sequence with a few transformer encoder layers.
    The final token is then used to output the weights and biases for each layer of the physics network.
    
    Physics network structure (parameters generated):
      - Layer 1: Linear(in_features=1, out_features=physnet_hidden)
      - Layer 2: Linear(physnet_hidden, physnet_hidden)
      - Layer 3: Linear(physnet_hidden, physnet_hidden)
      - Layer 4: Linear(physnet_hidden, physnet_hidden)
      - Output Layer: Linear(physnet_hidden, 3)
    """
    def __init__(
            self, seq_len=1000, input_dim=4, embed_dim=128, num_layers=2,
            num_heads=4, physnet_hidden=64, num_tokens=1, attention_learn=False
        ):
        super(HyperNet, self).__init__()
        self.seq_len = seq_len #nb of tokens, context
        self.embed_dim = embed_dim #dimension of the space the token is projected
        self.physnet_hidden = physnet_hidden #nb of neuron per layer in physnet
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_tokens = num_tokens
        # --- Embedding Stage ---
        # Embed each spatiotemporal point (e.g., [x, y, z, t]) into a higher-dimensional space.
        self.input_embed = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding (fixed, sinusoidal)
        #stored as non trainable parameters, buffer in pytorch = persistant tensor 
        # buffer use to store running statistics (so not immutable)
        #creation of self.pos_encoding
        #so the only thing i need to do to have unifexed nb of input is to put this in forward ?
        self.register_buffer("pos_encoding", self._generate_pos_encoding(self.seq_len, self.embed_dim))
        
        # Learnable token to aggregate information
        self.num_tokens = num_tokens  # define the number of tokens as a hyperparameter
        self.cls_tokens = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))
        
        if attention_learn:
            self.attn_layer = nn.Linear(embed_dim, 1)
        else: self.attn_layer = None

        # --- Transformer Encoder ---
        # We use a standard TransformerEncoder (with a few layers of multi‑head self-attention,
        # feed‑forward, normalization, and residual connections).
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   batch_first=True,
                                                   dim_feedforward=embed_dim * 4, #4 different than input dim
                                                   dropout=0.1,
                                                   activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- Output Heads for Physics Network Parameters ---
        # For each layer of the physics network we generate weight and bias.
        # Physics network architecture:
        #  * Layer 1: weight shape (physnet_hidden, 1), bias shape (physnet_hidden)
        #  * Layer 2: weight shape (physnet_hidden, physnet_hidden), bias shape (physnet_hidden)
        #  * Layer 3: weight shape (physnet_hidden, physnet_hidden), bias shape (physnet_hidden)
        #  * Layer 4: weight shape (physnet_hidden, physnet_hidden), bias shape (physnet_hidden)
        #  * Output layer: weight shape (3, physnet_hidden), bias shape (3)

        # Layer 1 heads
        self.gen_l1_weight = nn.Linear(embed_dim, physnet_hidden * 1)
        self.gen_l1_bias   = nn.Linear(embed_dim, physnet_hidden)
        
        # Layer 2 heads
        self.gen_l2_weight = nn.Linear(embed_dim, physnet_hidden * physnet_hidden)
        self.gen_l2_bias   = nn.Linear(embed_dim, physnet_hidden)
        
        # Layer 3 heads
        self.gen_l3_weight = nn.Linear(embed_dim, physnet_hidden * physnet_hidden)
        self.gen_l3_bias   = nn.Linear(embed_dim, physnet_hidden)
        
        # Layer 4 heads
        self.gen_l4_weight = nn.Linear(embed_dim, physnet_hidden * physnet_hidden)
        self.gen_l4_bias   = nn.Linear(embed_dim, physnet_hidden)
        
        # Output layer heads
        self.gen_out_weight = nn.Linear(embed_dim, 3 * physnet_hidden)
        self.gen_out_bias   = nn.Linear(embed_dim, 3)
        
    def _generate_pos_encoding(self, seq_len, d_model):
        """
        Generates a sinusoidal positional encoding.
        Returns a tensor of shape (1, seq_len, d_model).
        (d_model = embeded dim)
        each token define a position , and this position is split for different
        frequencies along the embeding dimension (will associate a diff freq to each dim)
        """
        #start with tensor of 0 size 1000x128
        pe = torch.zeros(seq_len, d_model)
        #create 2 d tensor colomn (each is numerator)
        #unsqueeze to allow element wise multiplication from (seq_len,) to seq_len,1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        #denominator to avoid too big number and stabilize freq (exp decay
        #smoothly to decrease the freq as d_model increase) (spread out fred)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) \
                * (-math.log(10000.0) / d_model)
        )
        #even indexes go sin
        pe[:, 0::2] = torch.sin(position * div_term)
        #odd indexes go cos
        pe[:, 1::2] = torch.cos(position * div_term)
        #unsqueeze to make it broadcastable
        pe = pe.unsqueeze(0)  #  from (seq_len, d_model) to (1, seq_len, d_model)
        return pe

    def forward(self, x):#x is context
        """
        x: Tensor of shape (B, seq_len, 4) where each point is [x, y, z, t]
        Returns a dictionary of generated parameters for the physics network.
        """
        B, L, _ = x.size()
        # Embed the input points
        x_emb = self.input_embed(x)  # (B, L, embed_dim)
        # Add fixed positional encoding (assumes L == self.seq_len and that
        #pos_encoding is broadcastable (1->B)
        x_emb = x_emb + self.pos_encoding
        # Prepend the learnable token to aggregate context (just expand to B)
        cls_tokens = self.cls_tokens.expand(B, -1, -1)  # (B, num_tokens, embed_dim)
        # add the tokens alogn the 1st dimension (sq_length)
        x_emb = torch.cat((cls_tokens, x_emb), dim=1)  # now the sequence length becomes (num_tokens + L)

        # Transformer encoder expects shape (S, B, D)
        x_emb = x_emb.transpose(0, 1)  # (num_tokens+L, B, embed_dim)
        transformer_out = self.transformer(x_emb)  # (num_tokens+L, B, embed_dim)
        # Transpose transformer output back to (B, num_tokens+L, embed_dim)
        transformer_out = transformer_out.transpose(0, 1)  # shape: (B, num_tokens+L, embed_dim)
        # Extract the learnable tokens (the first self.num_tokens tokens)
        token_out = transformer_out[:, :self.num_tokens, :]  # shape: (B, num_tokens, embed_dim)
        if self.attn_layer == None:
            summary = token_out.mean(dim=1)  # (B, embed_dim)
        else:
            # Aggregate the tokens to form a summary representation (e.g., by mean pooling)
            # Extract only the learnable tokens
            
            # --- Attention Pooling ---
            # Compute attention scores for each token
            attn_scores = self.attn_layer(token_out)  # (B, num_tokens, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)  # (B, num_tokens, 1)
            # Weighted sum to get the aggregated summary
            summary = (token_out * attn_weights).sum(dim=1)  # (B, embed_dim)
        
        
                
        # --- Generate physics network parameters from the summary ---
        params = {} #keys : 'l1_weight' , 'l1_bias' ect
        # Layer 1
        l1_weight = self.gen_l1_weight(summary)  # (B, physnet_hidden * 1)
        l1_bias   = self.gen_l1_bias(summary)      # (B, physnet_hidden)
        # .view is like reshape
        params['l1_weight'] = l1_weight.view(B, self.physnet_hidden, 1)
        params['l1_bias']   = l1_bias.view(B, self.physnet_hidden)
        
        # Layer 2
        l2_weight = self.gen_l2_weight(summary)  # (B, physnet_hidden * physnet_hidden)
        l2_bias   = self.gen_l2_bias(summary)      # (B, physnet_hidden)
        params['l2_weight'] = l2_weight.view(B, self.physnet_hidden, self.physnet_hidden)
        params['l2_bias']   = l2_bias.view(B, self.physnet_hidden)
        
        # Layer 3
        l3_weight = self.gen_l3_weight(summary)
        l3_bias   = self.gen_l3_bias(summary)
        params['l3_weight'] = l3_weight.view(B, self.physnet_hidden, self.physnet_hidden)
        params['l3_bias']   = l3_bias.view(B, self.physnet_hidden)
        
        # Layer 4
        l4_weight = self.gen_l4_weight(summary)
        l4_bias   = self.gen_l4_bias(summary)
        params['l4_weight'] = l4_weight.view(B, self.physnet_hidden, self.physnet_hidden)
        params['l4_bias']   = l4_bias.view(B, self.physnet_hidden)
        
        # Output layer
        out_weight = self.gen_out_weight(summary)
        out_bias   = self.gen_out_bias(summary)
        params['out_weight'] = out_weight.view(B, 3, self.physnet_hidden)
        params['out_bias']   = out_bias.view(B, 3)
        
        return params


class PhysNet(nn.Module):
    """
    A dynamic physics network (MLP) whose parameters are generated on the fly
    by the hypernetwork. This network maps a scalar time t to a 3D output [x, y, z].
    Architecture:
      - Layer 1: Linear(1 -> physnet_hidden)
      - Layers 2-4: Linear(physnet_hidden -> physnet_hidden)
      - Output layer: Linear(physnet_hidden -> 3)
    Each linear layer is followed by a GELU activation and LayerNorm.
    """
    def __init__(self, physnet_hidden=64):
        super(PhysNet, self).__init__()
        self.physnet_hidden = physnet_hidden
        # Predefine LayerNorm modules for consistency and efficiency
        self.norm1 = nn.LayerNorm(physnet_hidden)
        self.norm2 = nn.LayerNorm(physnet_hidden)
        self.norm3 = nn.LayerNorm(physnet_hidden)
        self.norm4 = nn.LayerNorm(physnet_hidden)
        self.parameters = None

    def set_parameters(self, parameters):
        self.parameters = parameters

    def forward(self, t, params):
        # Layer 1: (1 -> physnet_hidden)
        x = API.tools.batched_linear(t, params['l1_weight'], params['l1_bias'])
        x = F.gelu(x)
        x = self.norm1(x)
        
        # Layer 2: (physnet_hidden -> physnet_hidden)
        x = API.tools.batched_linear(x, params['l2_weight'], params['l2_bias'])
        x = F.gelu(x)
        x = self.norm2(x)
        
        # Layer 3: (physnet_hidden -> physnet_hidden)
        x = API.tools.batched_linear(x, params['l3_weight'], params['l3_bias'])
        x = F.gelu(x)
        x = self.norm3(x)
        
        # Layer 4: (physnet_hidden -> physnet_hidden)
        x = API.tools.batched_linear(x, params['l4_weight'], params['l4_bias'])
        x = F.gelu(x)
        x = self.norm4(x)
        
        # Output layer: (physnet_hidden -> 3)
        out = API.tools.batched_linear(x, params['out_weight'], params['out_bias'])
        return out
    
class Coach:
    """
    Trainer class for joint training of the hypernetwork (which generates the physics network parameters)
    and the physics network itself, with evaluation on test data to monitor overfitting.
    """
    def __init__( self, model, dataBase, device, resolution=2, lr=1e-3, pde_weight=0.1):
        """
        Parameters:
          - model: An instance of DeepPhysiNet.
          - train_loader: DataLoader for the training set.
          - test_within_loader: DataLoader for trajectories from training zones.
          - test_unseen_loader: DataLoader for trajectories from unseen zones.
          - loss_fn: HybridLoss (combines data and physics loss).
          - optimizer: Optimizer (e.g., Adam) updating model parameters.
          - device: Torch device ('cuda' or 'cpu').
        """
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.device = device
        self.dataBase = dataBase
        self.t_min = dataBase.t_min
        self.t_max = dataBase.t_max
        self.scaler_state = dataBase.scaler_state
        self.resolution = resolution
        #hybridloss to device ?
        self.loss_fn = API.HybridLoss(
            t_min=self.t_min,t_max=self.t_max, 
            scaler_state=self.scaler_state,
            pde_weight=pde_weight
        ).to(self.device)
        #should .to_device(devide) this ?
        self.train_loader = self.dataBase.get_train_loader(shuffle=True)
        self.test_within_loader = self.dataBase.get_test_within_loader(shuffle=False)
        self.test_unseen_loader = self.dataBase.get_test_unseen_loader(shuffle=False)
        print("Dataset loaded")
        # Initialize TensorBoard SummaryWriter.
        # We use the same API.config.LOGS_DIR used for saving plots, appending a subfolder.
        log_dir = Path(API.config.LOGS_DIR) / "tensorboard" / self.get_name()
        log_dir.mkdir(parents=True, exist_ok=True)  # create directory if it doesn't exist
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def evaluate(self, loader, high_res_eval=False):
        #set to false for training , not testing mode
        """Evaluate the model on a given data loader.
        If high_res_eval is True, use the full high-res half (i.e. full_time/full_state).
        Otherwise, use only the grid points.
        Only the data (MSE) loss is computed.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                context = batch['context'].to(self.device)
                params = self.model.hypernet(context)
                
                if high_res_eval:
                    full_time = batch['full_time'].to(self.device)      # (B, high_res_length, 1)
                    full_state = batch['full_state'].to(self.device)    # (B, high_res_length, 3)
                    pred_full = self.model.physnet(full_time, params)
                    # Here we skip PDE loss.
                    loss, _, _ = self.loss_fn(
                        t_norm=full_time,
                        space_pred_scaled=pred_full,
                        space_true=full_state,
                        compute_data_loss=True, 
                        skip_pde=True
                    )
                else:
                    grid_time = batch['grid_time'].to(self.device)      # (B, context_tokens, 1)
                    grid_state = batch['grid_state'].to(self.device)    # (B, context_tokens, 3)
                    pred = self.model.physnet(grid_time, params)
                    loss, _, _ = self.loss_fn(
                        t_norm=grid_time,
                        space_pred_scaled=pred,
                        space_true=grid_state,
                        compute_data_loss=True,
                        skip_pde=True
                    )
                        
                total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        self.model.train()
        return avg_loss


    #need to test on unsee resolution also 
    def train(self, num_epochs=100):
        print("Start training")
    
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                
                # Retrieve precomputed splits.
                context   = batch['context'].to(self.device)      # (B, context_tokens, 4)
                grid_time = batch['grid_time'].to(self.device)      # (B, context_tokens, 1)
                #grid_state= batch['grid_state'].to(self.device)     # (B, context_tokens, 3)
                inner_time = batch['inner_time'].to(self.device)      # (B, inner_count, 1)
                #denorm_grid_time = batch['denorm_grid_time'].to(self.device)      # (B, context_tokens, 1)
                grid_state = batch['grid_state'].to(self.device)     # (B, context_tokens, 3)
                #denorm_inner_time= batch['denorm_inner_time'].to(self.device)      # (B, inner_count, 1)



                # 3) We need gradients on the *real* times for PDE
                #not grid_time = grid_time.require_grad() ?
                grid_time.requires_grad_()
                inner_time.requires_grad_()
                    # Generate dynamic parameters from context.

                params = self.model.hypernet(context)
                
                # Predictions.
                pred_grid  = self.model.physnet(grid_time, params)   # (B, context_tokens, 3)
                pred_inner = self.model.physnet(inner_time, params)   # (B, inner_count, 3)
                
                # Compute losses., denormtime useless here
                #donc c'est que de ici que normalisation de x y z depend ?
                loss_grid, data_loss, pde_loss_grid = self.loss_fn(
                    t_norm=grid_time, 
                    space_pred_scaled=pred_grid, 
                    space_true=grid_state, 
                    compute_data_loss=True
                )
                #only pde so need denorm time for pde
                loss_inner, _, pde_loss_inner = self.loss_fn(
                    t_norm=inner_time, 
                    space_pred_scaled=pred_inner, 
                    space_true=None, 
                    compute_data_loss=False
                )

                total_loss = loss_grid + loss_inner
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)#+15pp
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
            
            avg_train_loss = epoch_loss / len(self.train_loader)
            test_within_loss = self.evaluate(
                self.test_within_loader,
                high_res_eval=True
            )
            test_unseen_loss = self.evaluate(
                self.test_unseen_loader, 
                high_res_eval=True
            )
            
            self.writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            self.writer.add_scalar('Loss/Test_Within', test_within_loss, epoch)
            self.writer.add_scalar('Loss/Test_Unseen', test_unseen_loss, epoch)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.6f} | Test Within Loss: {test_within_loss:.6f} | Test Unseen Loss: {test_unseen_loss:.6f}")
                # At the very end of your train() method, after the epoch loop:
        save_path = Path(API.config.MODELS_DIR) / (self.get_name() + ".pth")
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        # Optionally call final_evaluate() after training.

    def inverse_time_norm(self, t_norm):
        """
        Convert normalized time t_norm back to real time 
        using min-max scaling with self.t_min, self.t_max.
        """
        return t_norm * (self.t_max - self.t_min) + self.t_min

    def get_name(self):

        #contextSize_embededDim_nbLayers_nbHeadsPerLayer_nbTrainableTokens_resolution_contextFraction
        name = "cS" + str(self.model.hypernet.seq_len) + "_eD" + str(self.model.hypernet.embed_dim) + \
            "_nL" + str(self.model.hypernet.num_layers) + "_nH" + str(self.model.hypernet.num_heads) \
            + "_nT" + str(self.model.hypernet.num_tokens) + "_rl" + str(self.resolution) \
            + "_cF" + str(self.dataBase.context_fraction) + "_dS" + str(self.dataBase.load_size)
        return name

    def plot_loss_curves(
            self, train_losses, test_within_losses,
              test_unseen_losses, save=False, filepath=API.config.FIG_DIR):
        """
        Plots the training loss, test within loss, and test unseen loss curves over epochs.
        """
        filepath = Path(filepath) / (self.get_name() + ".png")
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-', label='Train Loss')
        plt.plot(epochs, test_within_losses, 'r-', label='Test Within Loss')
        plt.plot(epochs, test_unseen_losses, 'g-', label='Test Unseen Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss Curves')
        plt.legend()
        plt.grid(True)
        if save:
            plt.savefig(filepath)
        plt.show()

class HybridLoss(nn.Module):
    def __init__(
            self, 
            sigma=10.0, rho=28.0, beta=8.0/3.0,
            t_min=0.0, t_max=100.0,
            scaler_state=None,
            pde_weight=0.1
    ):
        """
        scaler_state: your fitted StandardScaler for (x,y,z).
                      We'll use scaler_state.mean_ and scaler_state.scale_
                      to un-scale PDE derivatives.
        """
        super(HybridLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        #keep in cpu bc onlu for small float
        self.t_min = t_min
        self.t_max = t_max
        self.pde_weight = pde_weight
        
        self.scaler_state = scaler_state  # pass the same scaler used in DataBase
        if self.scaler_state is not None:
            # We'll store them as torch tensors in forward(...) to match device
            self.mean_ = None
            self.scale_ = None

    def forward(self, t_norm, space_pred_scaled, space_true=None,
                compute_data_loss=True, skip_pde=False):
        """
        t_norm: shape (B, T, 1) in [0..1].
        space_pred_scaled: shape (B, T, 3), net output in scaled domain if scaler is used.
        space_true: shape (B, T, 3) => can be scaled or unscaled, your choice.
        skip_pde: if True, skip PDE constraints.

        We'll do chain rule for both time & state, to get real derivatives from scaled outputs.
        """
        #should not space.requires_grad before ?
        if compute_data_loss and space_true is not None:
            data_loss = self.mse_loss(space_pred_scaled, space_true)
        else:
            data_loss = 0.0

        # 2) If skip_pde => just return data_loss
        if skip_pde:
            return data_loss, data_loss, 0.0

        # 3) PDE chain rule for time
        chain_factor_time = 1.0 / (self.t_max - self.t_min)

        # 4) Enable gradient on space_pred_scaled wrt t_norm
        space_pred_scaled.requires_grad_(True)
        #or not the = ? 
        # For each dimension x,y,z => d(...) / d(t_norm)
        dx_dtau = torch.autograd.grad(
            outputs=space_pred_scaled[..., 0],  # x_scaled
            inputs=t_norm,
            grad_outputs=torch.ones_like(space_pred_scaled[..., 0]),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        dy_dtau = torch.autograd.grad(
            outputs=space_pred_scaled[..., 1],  # y_scaled
            inputs=t_norm,
            grad_outputs=torch.ones_like(space_pred_scaled[..., 1]),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        dz_dtau = torch.autograd.grad(
            outputs=space_pred_scaled[..., 2],  # z_scaled
            inputs=t_norm,
            grad_outputs=torch.ones_like(space_pred_scaled[..., 2]),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Squeeze final dim if shape is (B,T,1)
        dx_dtau = dx_dtau.squeeze(-1)
        dy_dtau = dy_dtau.squeeze(-1)
        dz_dtau = dz_dtau.squeeze(-1)

        # 5) Also chain rule for state scaling
        # We'll get scale_x, scale_y, scale_z from self.scaler_state
        if self.mean_ is None or self.scale_ is None:
            # create them as torch tensors on correct device
            self.mean_ = torch.tensor(self.scaler_state.mean_, dtype=torch.float32,
                                      device=t_norm.device).view(1,1,3)
            self.scale_ = torch.tensor(self.scaler_state.scale_, dtype=torch.float32,
                                       device=t_norm.device).view(1,1,3)
        
        # scale_ => shape (1,1,3). We'll do dimension wise multiplication

        # dx_dtau (B,T), but we want to multiply by scale_ for x => scale_[...,0], etc.
        # Let's just gather them carefully. We can do this by indexing or by a gather approach.

        # dxdt_pred_real = dx_dtau_scaled * scale_x * chain_factor_time
        dxdt_pred_real = dx_dtau * self.scale_[...,0] * chain_factor_time
        dydt_pred_real = dy_dtau * self.scale_[...,1] * chain_factor_time
        dzdt_pred_real = dz_dtau * self.scale_[...,2] * chain_factor_time

        # 6) Now convert scaled states => real
        x_scaled = space_pred_scaled[...,0]
        y_scaled = space_pred_scaled[...,1]
        z_scaled = space_pred_scaled[...,2]
        # x_real = x_scaled * scale_x + mean_x
        x_real = x_scaled * self.scale_[...,0] + self.mean_[...,0]
        y_real = y_scaled * self.scale_[...,1] + self.mean_[...,1]
        z_real = z_scaled * self.scale_[...,2] + self.mean_[...,2]

        # 7) PDE in real scale
        dxdt_lorenz = self.sigma * (y_real - x_real)
        dydt_lorenz = x_real * (self.rho - z_real) - y_real
        dzdt_lorenz = x_real * y_real - self.beta * z_real

        # 8) PDE MSE
        physics_loss_x = self.mse_loss(dxdt_pred_real, dxdt_lorenz)
        physics_loss_y = self.mse_loss(dydt_pred_real, dydt_lorenz)
        physics_loss_z = self.mse_loss(dzdt_pred_real, dzdt_lorenz)
        physics_loss = physics_loss_x + physics_loss_y + physics_loss_z

        total_loss = data_loss + self.pde_weight*physics_loss
        return total_loss, data_loss, physics_loss
