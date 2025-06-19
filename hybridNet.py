
import meteo_api as API
import matplotlib.pyplot as plt # type: ignore
import math
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore
from pathlib import Path


class HybridNet(nn.Module):
    def __init__(

        ):
        pass
       
    def forward(self):
        pass


class Coach:

    def __init__( self, model, dataBase, device, lr=1e-3):
       
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.device = device
        self.dataBase = dataBase
        
        self.loss_fn = API.HybridLoss(
        ).to(self.device)
        
        log_dir = Path(API.config.LOGS_DIR) / "tensorboard" / self.get_name()
        log_dir.mkdir(parents=True, exist_ok=True)  # create directory if it doesn't exist
        self.writer = SummaryWriter(log_dir=str(log_dir))


    #need to test on unsee resolution also 
    def train(self, num_epochs=100):
        pass

    

class HybridLoss(nn.Module):
    def __init__(
            self
    ):
        pass

    def forward():
        pass
