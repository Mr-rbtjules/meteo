
import meteo_api as API


import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore

class PhysNet(nn.Module):
    def __init__(
            self, input_dim=1, hidden_dim=64, 
            output_dim=3, num_hidden_layers=3
        ):
        super(PhysNet, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, t):
        return self.model(t)

class CombinedLoss(nn.Module):
    def __init__(self, scaler_y, sigma=10.0, rho=28.0, beta=8.0/3.0):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.scaler_y = scaler_y  # To inverse transform if necessary
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
    
    def forward(self, t, y_pred, y_true):
        # Data Loss
        data_loss = self.mse_loss(y_pred, y_true)
        
        # Compute derivatives using autograd
        y_pred = y_pred.requires_grad_(True)
        dydt = torch.autograd.grad(
            outputs=y_pred,
            inputs=t,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]  # Shape: (batch_size, 3)
        
        # Extract individual components
        x_pred = y_pred[:, 0]
        y_pred_var = y_pred[:, 1]
        z_pred = y_pred[:, 2]
        
        dxdt_pred = dydt[:, 0]
        dydt_pred = dydt[:, 1]
        dzdt_pred = dydt[:, 2]
        
        # Define Lorenz ODEs
        dxdt_lorenz = self.sigma * (y_pred_var - x_pred)
        dydt_lorenz = x_pred * (self.rho - z_pred) - y_pred_var
        dzdt_lorenz = x_pred * y_pred_var - self.beta * z_pred
        
        # Physics Loss
        physics_loss_x = self.mse_loss(dxdt_pred, dxdt_lorenz)
        physics_loss_y = self.mse_loss(dydt_pred, dydt_lorenz)
        physics_loss_z = self.mse_loss(dzdt_pred, dzdt_lorenz)
        
        physics_loss = physics_loss_x + physics_loss_y + physics_loss_z
        
        # Total Loss
        total_loss = data_loss + physics_loss
        return total_loss, data_loss, physics_loss
