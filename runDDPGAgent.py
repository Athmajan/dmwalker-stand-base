import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dm_control import suite
from torch.distributions import Normal
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import time
import random
from torch.autograd import Variable
from collections import deque, OrderedDict
import cv2



from typing import List, Tuple, OrderedDict
from agent import Agent
from agent_ddpg import DDPGAgent, TD3



'''
This can be used to test the functionality 
of the DDPG base policy agent

Given an observation the agent gives the action
based on the pre trained model

'''

domainName = "walker" # Name of a environment (set it to any Continous environment you want)
taskName = "stand" # Name of a environment (set it to any Continous environment you want)
env_name = domainName+ "_"+taskName
seed = 0 # Random seed number
file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = suite.load(domain_name=domainName, task_name=taskName, task_kwargs={'random': seed})

obsSpec = env.observation_spec()
action_spec = env.action_spec()

orientDim = obsSpec['orientations'].shape[0]
heightDim = len(obsSpec['height'].shape) + 1 
velocityDim = obsSpec['velocity'].shape[0]
input_dim = orientDim + heightDim + velocityDim

hidden_dim = 128
output_dim = env.action_spec().shape[0]


state_dim = input_dim
action_dim = output_dim
max_action = float(1)

# Initialize TD3 with the device
policy = TD3(state_dim, action_dim, max_action, device)
policy.load(file_name, directory="./pytorch_models")

# Define bin edges for discretization
discretize = True
num_bins = 20  # 20 bins from -1 to 1


# Initialize DDPGAgent with the policy and device
agent = DDPGAgent(agent_id=1, num_bins=num_bins, policy=policy, device=device)
eval_episodes=10

def process_state(state):
    if isinstance(state, OrderedDict):
        if 'orientations' in state and 'height' in state and 'velocity' in state:
            orient = state['orientations']
            height = state['height']
            velocity = state['velocity']
            if np.isscalar(height):
                height = np.array([height])
            out = np.concatenate((orient, height, velocity))
            return out
    elif isinstance(state, np.ndarray) and state.shape == (24,):
        return state
    elif hasattr(state, 'observation') and isinstance(state.observation, OrderedDict):
        observation = state.observation
        if 'orientations' in observation and 'height' in observation and 'velocity' in observation:
            orient = observation['orientations']
            height = observation['height']
            velocity = observation['velocity']
            if np.isscalar(height):
                height = np.array([height])
            out = np.concatenate((orient, height, velocity))
            return out
    else:
        raise ValueError("Input state must be either an OrderedDict with keys 'orientations', 'height', and 'velocity', a numpy ndarray of shape (24,), or a TimeStep object with a valid observation.")




avg_reward = 0.
state = env.reset()
obs = process_state(state)
done = False
while not done:
    bestHalf, best = agent.act(obs)
    print(agent.id)
    print(bestHalf)
    print(best)

    # state = env.step(action)
    # obs = process_state(state)
    # reward = state.reward
    # done = state.last()
    break



# m_agents = 2

# agents = [
#             DDPGAgent(
#                 i, 20,policy, device
#             ) for i in range(m_agents)
#         ]

# for agent in agents:
#     print(agent.id)