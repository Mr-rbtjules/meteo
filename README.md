# Physics-Informed Machine Learning for Dynamical Systems & Weather Forecasting

Embedding physical laws directly into neural network training for learning and predicting chaotic dynamical systems — from the classical Lorenz-63 attractor to the higher-dimensional Lorenz-96 weather model.

---

## Overview

This project explores **physics-informed machine learning** approaches where known governing equations (ODEs) are enforced as soft constraints during training. Instead of relying purely on data, the models learn dynamics that are consistent with the underlying physics, improving generalization and physical plausibility.

Two complementary systems are implemented across two branches:

| Branch | System | Models | Key Idea |
|--------|--------|--------|----------|
| `main` | **Lorenz-63** (3D chaotic attractor) | HyperNetwork (Transformer) + PhysNet (MLP) | Meta-learning: a Transformer generates MLP weights from context trajectories |
| `secondTry` | **Lorenz-96** (10D weather model) | Physics-Informed LSTM / Encoder-Decoder Transformer | Partial observability: reconstruct full state from 5 of 10 variables |

---

## Architecture

### Branch `main` — HyperNetwork for Lorenz-63

The core model is a **HybridNet = HyperNet + PhysNet**:

- **HyperNet** (Transformer Encoder): Ingests a spatiotemporal context sequence `[x, y, z, t]`, uses a learnable CLS token, and outputs all weights & biases for the PhysNet
- **PhysNet** (4-layer MLP): Takes scalar time `t` → predicts state `[x, y, z]`. Its parameters are **dynamically generated** by the HyperNet per sample

```
Context trajectory → [Transformer Encoder] → CLS token → [Linear Heads] → PhysNet weights
                                                                              ↓
                                              Scalar time t → [PhysNet (MLP)] → [x, y, z]
```

### Branch `secondTry` — LSTM & Transformer for Lorenz-96

Two architectures tackle the **partial observation** problem (observing 5 of 10 state variables):

- **PILSTMNet**: Standard LSTM with closed-loop prediction (predictions fed back as inputs)
- **TransformerNet**: Custom encoder-decoder Transformer with causal masking and teacher forcing

---

## Physics-Informed Loss

The loss function enforces the **Lorenz equations** as a differentiable constraint:

**Lorenz-63** (`main`):
```
dx/dt = σ(y − x)        σ = 10
dy/dt = x(ρ − z) − y    ρ = 28
dz/dt = xy − βz         β = 8/3
```

**Lorenz-96** (`secondTry`):
```
dy_i/dt = (y_{i+1} − y_{i−2}) · y_{i−1} − y_i + F    F = 8
```

The total loss combines:
- **Data loss**: MSE between predictions and ground truth at observed points
- **PDE loss**: MSE between autograd-computed derivatives and the ODE right-hand side

A key feature: the physics loss can be evaluated at **arbitrary points without labels**, enabling supervision beyond the training data.

The implementation carefully handles chain rules through both time normalization (min-max to [0,1]) and state normalization (StandardScaler).

---

## Data Generation

- **Lorenz-63**: High-precision RK4 integrator (h=1e-5) with Numba JIT compilation. 100 initial condition zones, 100 trajectories each, stored in HDF5
- **Lorenz-96**: Euler integration of a single long trajectory (20,000 steps, dt=0.01), with sliding-window segmentation for batching

---

## Project Structure

```
meteo_api/
├── physicalNetwork.py    # HybridNet, HyperNet, PhysNet, Coach, HybridLoss (main)
├── hybridNet.py          # PILSTMNet, Coach, HybridLoss for Lorenz-96 (secondTry)
├── transformerNet.py     # Encoder-Decoder Transformer for Lorenz-96 (secondTry)
├── dataBase.py           # Dataset classes, HDF5 loading, normalization
├── tools.py              # Lorenz simulation (RK4/Numba), plotting, animation
├── config.py             # Hyperparameters and paths
└── __init__.py           # Package exports
```

---

## Getting Started

### Requirements

```
torch
numpy
numba
h5py
scikit-learn
matplotlib
joblib
tensorboard
```

### Usage

```python
import meteo_api as API
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate Lorenz-63 trajectory data
API.TrajectoryDataset.createDbFromScratch(
    API.config.RAW_DATA_DIR + "/trajectories_dataset/", zones=100
)

# Load data
db = API.DataBase(context_fraction=0.5, context_tokens=100, load_size=0.1, resolution=2)

# Build model
model = API.HybridNet(
    seq_len=100, embed_dim=128, num_layers=2,
    num_heads=4, physnet_hidden=64, num_tokens=1
)

# Train with physics-informed loss
coach = API.Coach(model, db, device, resolution=2, lr=1e-3, pde_weight=0.1)
coach.train(num_epochs=100)
```

Monitor training:
```bash
tensorboard --logdir=data/logs/tensorboard/
```

---

## References

- **Elise Ozalp, Georgios Margazoglou, Luca Magri** — *Physics-Informed Long Short-Term Memory for Forecasting and Reconstruction of Chaos* ([arXiv:2302.10779](https://arxiv.org/abs/2302.10779))
