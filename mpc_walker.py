from casadi import *
import torch
import numpy as np

# Load the PyTorch model
model = torch.load('/Users/athmajanvivekananthan/WCE/JEPA - MARL/multi-agent/dmcontrol/dmWalker/artifacts/990_walkerDynamics.pt')
for layer in model:
    print(layer)

