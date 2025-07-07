
import os
import random
import numpy as np
import torch



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "./data/logs")
MODELS_DIR = os.path.join(BASE_DIR, "./data/saved_models")
RAW_DATA_DIR = os.path.join(BASE_DIR, "./data/lorenzData")
FIG_DIR = os.path.join(BASE_DIR, "./data/figures")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "./data/saved_models") # Directory to save trained models

# Learning Rate Scheduler Parameters
LR_SCHEDULER_PATIENCE = 10 # Number of epochs with no improvement after which learning rate will be reduced
LR_SCHEDULER_FACTOR = 0.5 # Factor by which the learning rate will be reduced. new_lr = lr * factor






SEED_NP = 4
SEED_RAND = 5
SEED_SHUFFLE = 6
SEED_TORCH = 7
TEST_DATA_PROPORTION = 0.2
BATCH_SIZE = 64 # Batch size for training and validation
#more than one if non trivial computation in getitem
NUM_WORKERS = 0 #main process


M = 10e4

# Lorenz-96 Parameters
N_LORENZ = 10 # Total number of system variables
F_LORENZ = 8 # Forcing parameter
DT_LORENZ = 0.01 # Time step for simulation
NUM_STEPS_LORENZ = 20000 # Number of timesteps for the simulated trajectory
NX_LORENZ = 5 # Number of observed variables

# LSTM Parameters
LSTM_HIDDEN_SIZE = 100
LSTM_NUM_LAYERS = 1
LSTM_DROPOUT_RATE = 0.0 # Set dropout to 0 for single-layer LSTM to avoid warning

ALPHA_PI = 1e-2 # Weight for physics-informed loss
SEGMENT_LENGTH = 200 # Length of trajectory segments for batching = number of sliding windows returned by getitem

# Transformer Parameters
TRANSFORMER_NUM_ENCODER_LAYERS = 2
TRANSFORMER_NUM_DECODER_LAYERS = 2
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_MODEL_DIM = 28
TRANSFORMER_FF_DIM = 112
TRANSFORMER_DROPOUT_RATE = 0.1

# Global Model Configuration
MODEL_TYPE = 'PILSTM' # 'Transformer' or 'PILSTM'
