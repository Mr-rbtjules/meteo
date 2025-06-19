
from .config import SEED_NP
import numpy as np
from numba import njit  # type: ignore
import os
import h5py  # type: ignore
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt # type: ignore
import torch # type: ignore
import meteo_api as API


###LORENZ SIMULATION###
cache = False
@njit(cache=cache)
def lorenz():
   return None
