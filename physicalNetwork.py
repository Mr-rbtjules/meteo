
import meteo_api as API

import math
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore


class DeepPhysiNet(nn.Module):
    def __init__(self, seq_len=1000, input_dim=4, embed_dim=128,
                 num_layers=2, num_heads=4, physnet_hidden=64, num_tokens=1):
        super(DeepPhysiNet, self).__init__()
        self.hypernet = HyperNet(seq_len=seq_len,
                                 input_dim=input_dim,
                                 embed_dim=embed_dim,
                                 num_layers=num_layers,
                                 num_heads=num_heads,
                                 physnet_hidden=physnet_hidden,
                                 num_tokens=num_tokens)  # pass num_tokens if you decide to use more
        self.physnet = PhysNet(physnet_hidden=physnet_hidden)
    
    def forward(self, context, t):
        # Generate physics network parameters from context using the hypernetwork
        params = self.hypernet(context)
        # Compute the prediction at time t using the dynamic physics network
        pred = self.physnet(t, params)
        return pred


#Hypernetwork class


##############################################
#   HyperNetwork and PhysNet for Lorenz Toy    #
##############################################

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

class HyperNet(nn.Module):
    """
    A transformer‐based hyper‑network that generates the parameters for the physics network.
    
    The input to the hyper‑net is a batch of spatiotemporal context points (e.g., 1000 points,
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
    def __init__(self, seq_len=1000, input_dim=4, embed_dim=128, num_layers=2, num_heads=4, physnet_hidden=64, num_tokens=1):
        super(HyperNet, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.physnet_hidden = physnet_hidden
        
        # --- Embedding Stage ---
        # Embed each spatiotemporal point (e.g., [x, y, z, t]) into a higher-dimensional space.
        self.input_embed = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding (fixed, sinusoidal)
        self.register_buffer("pos_encoding", self._generate_pos_encoding(seq_len, embed_dim))
        
        # Learnable token to aggregate information
        self.num_tokens = num_tokens  # define the number of tokens as a hyperparameter
        self.cls_tokens = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))
        
        # --- Transformer Encoder ---
        # We use a standard TransformerEncoder (with a few layers of multi‑head self-attention,
        # feed‑forward, normalization, and residual connections).
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=embed_dim * 4,
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
        """
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        return pe

    def forward(self, x):#x is context
        """
        x: Tensor of shape (B, seq_len, 4) where each point is [x, y, z, t]
        Returns a dictionary of generated parameters for the physics network.
        """
        B, L, _ = x.size()
        # Embed the input points
        x_emb = self.input_embed(x)  # (B, L, embed_dim)
        # Add fixed positional encoding (assumes L == self.seq_len)
        x_emb = x_emb + self.pos_encoding[:, :L, :]
        # Prepend the learnable token to aggregate context
        cls_tokens = self.cls_tokens.expand(B, -1, -1)  # (B, num_tokens, embed_dim)
        x_emb = torch.cat((cls_tokens, x_emb), dim=1)  # now the sequence length becomes (num_tokens + L)
                
        # Transformer encoder expects shape (S, B, D)
        x_emb = x_emb.transpose(0, 1)  # (num_tokens+L, B, embed_dim)
        transformer_out = self.transformer(x_emb)  # (num_tokens+L, B, embed_dim)
        # Transpose transformer output back to (B, num_tokens+L, embed_dim)
        transformer_out = transformer_out.transpose(0, 1)  # shape: (B, num_tokens+L, embed_dim)
        # Extract the learnable tokens (the first self.num_tokens tokens)
        token_out = transformer_out[:, :self.num_tokens, :]  # shape: (B, num_tokens, embed_dim)
        # Aggregate the tokens to form a summary representation (e.g., by mean pooling)
        summary = token_out.mean(dim=1)  # (B, embed_dim)
                
        # --- Generate physics network parameters from the summary ---
        params = {} #keys : 'l1_weight' , 'l1_bias' ect
        # Layer 1
        l1_weight = self.gen_l1_weight(summary)  # (B, physnet_hidden * 1)
        l1_bias   = self.gen_l1_bias(summary)      # (B, physnet_hidden)
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
    
    def forward(self, t, params):
        # Layer 1: (1 -> physnet_hidden)
        x = batched_linear(t, params['l1_weight'], params['l1_bias'])
        x = F.gelu(x)
        x = self.norm1(x)
        
        # Layer 2: (physnet_hidden -> physnet_hidden)
        x = batched_linear(x, params['l2_weight'], params['l2_bias'])
        x = F.gelu(x)
        x = self.norm2(x)
        
        # Layer 3: (physnet_hidden -> physnet_hidden)
        x = batched_linear(x, params['l3_weight'], params['l3_bias'])
        x = F.gelu(x)
        x = self.norm3(x)
        
        # Layer 4: (physnet_hidden -> physnet_hidden)
        x = batched_linear(x, params['l4_weight'], params['l4_bias'])
        x = F.gelu(x)
        x = self.norm4(x)
        
        # Output layer: (physnet_hidden -> 3)
        out = batched_linear(x, params['out_weight'], params['out_bias'])
        return out
    


class HybridLoss(nn.Module):
    def __init__(self, scaler_y, sigma=10.0, rho=28.0, beta=8.0/3.0):
        super(HybridLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.scaler_y = scaler_y  # To inverse transform if necessary
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
    
    def forward(self, t, space_pred, space_true):
        #y = space coord x,y,z
        # Data Loss
        data_loss = self.mse_loss(space_pred, space_true)
        
        # Compute derivatives using autograd
        space_pred = space_pred.requires_grad_(True)
        dydt = torch.autograd.grad(
            outputs=space_pred,
            inputs=t,
            grad_outputs=torch.ones_like(space_pred), #seed for autodiff
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]  # Shape: (batch_size, 3)
        
        # Extract individual components
        x_pred = space_pred[:, 0]
        y_pred = space_pred[:, 1]
        z_pred = space_pred[:, 2]
        
        dxdt_pred = dydt[:, 0]
        dydt_pred = dydt[:, 1]
        dzdt_pred = dydt[:, 2]
        
        # Define Lorenz ODEs
        dxdt_lorenz = self.sigma * (y_pred - x_pred)
        dydt_lorenz = x_pred * (self.rho - z_pred) - y_pred
        dzdt_lorenz = x_pred * y_pred - self.beta * z_pred
        
        # Physics Loss
        physics_loss_x = self.mse_loss(dxdt_pred, dxdt_lorenz)
        physics_loss_y = self.mse_loss(dydt_pred, dydt_lorenz)
        physics_loss_z = self.mse_loss(dzdt_pred, dzdt_lorenz)
        
        physics_loss = physics_loss_x + physics_loss_y + physics_loss_z
        
        # Total Loss
        total_loss = data_loss + physics_loss
        return total_loss, data_loss, physics_loss
    

#still need training and save
##############################################
#  Example usage (you can integrate this into your training loop)
##############################################
# Suppose you have:
#   - context: a tensor of shape (B, 1000, 4) from your database (e.g. normalized [x,y,z,t])
#   - t: a tensor of shape (B, 1) for the query time points
#
# model = DeepPhysiNet()
# pred_xyz = model(context, t)
#
# Then, you can compute your loss with CombinedLoss and backpropagate accordingly.
#physnetclass
