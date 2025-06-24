
import os
import random
import numpy as np
import torch



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "./data/logs")
MODELS_DIR = os.path.join(BASE_DIR, "./data/saved_models")
RAW_DATA_DIR = os.path.join(BASE_DIR, "./data/lorenzData")
FIG_DIR = os.path.join(BASE_DIR, "./data/figures")






SEED_NP = 4
SEED_RAND = 5
SEED_SHUFFLE = 6
SEED_TORCH = 7
TEST_DATA_PROPORTION = 0.2
BATCH_SIZE = 32
#more than one if non trivial computation in getitem
NUM_WORKERS = 1 #more than 1 = slower wtf => keep low for low data
PHYSNET_HIDDEN = 32

M = 10e4

# Lorenz-96 Parameters
N_LORENZ = 10 # Total number of system variables
F_LORENZ = 8 # Forcing parameter
DT_LORENZ = 0.01 # Time step for simulation
NUM_STEPS_LORENZ = 20000 # Number of timesteps for the simulated trajectory
NX_LORENZ = 5 # Number of observed variables

# Transformer Parameters
SEQUENCE_LENGTH = 10 # Length of input sequence for Transformer
NUM_LAYERS = 2 # Number of Transformer decoder layers
NUM_HEADS = 4 # Number of attention heads
EMBEDDING_DIM = 64 # Dimension of the embedding space
ALPHA_PI = 1e-3 # Weight for physics-informed loss
SEGMENT_LENGTH = 200 # Length of trajectory segments for batching
