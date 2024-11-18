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

from agent import Agent
from typing import List, Dict, Tuple, OrderedDict



from tdmpc import TDMPC
from cfg import parse_cfg
from pathlib import Path
__CONFIG__, __LOGS__ = 'cfgs', 'logs'
domainName = "walker" # Name of a environment (set it to any Continous environment you want)
taskName = "stand" # Name of a environment (set it to any Continous environment you want)


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


# Function to visualize the environment
def visualize(env):
    frameA = np.hstack([env.physics.render(480, 480, camera_id=0),
                        env.physics.render(480, 480, camera_id=1)])
    return frameA


def writeMovie(frames, fileName):
    # Define the codec and create a VideoWriter object
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{fileName}.mp4', fourcc,100.0, (width, height))

    # Write the frames to the video file
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    # Release the video writer
    out.release()
    return 

if __name__ == "__main__":
    cfg = parse_cfg(Path().cwd() / __CONFIG__)
    cfg.obs_shape = tuple(int(x) for x in (24,))
    cfg.action_shape = tuple(int(x) for x in (6,))
    cfg.action_dim = 6
    mpc_agent = TDMPC(cfg)
    mpc_agent.load("/Users/athmajanvivekananthan/WCE/JEPA - MARL/multi-agent/tdmpc/tdmpc_athmajan/models_backup_Nov16/step_500000.pth")
    
    frames = []
    env = suite.load(domain_name=domainName, task_name=taskName)
    done = False
    state = env.reset()
    frames.append(visualize(env))
    t = 0
    obs = process_state(state)
    epi_reward = 0
    
    while not done:
        action = mpc_agent.plan(obs, eval_mode=True, step=10000, t0=t==0)
        state = env.step(action)
        obs = process_state(state)
        reward = state.reward
        done = state.last()
        epi_reward += reward
        t += 1
        print(t,epi_reward)
        frames.append(visualize(env))

    writeMovie(frames,"testMPC")




