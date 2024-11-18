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
from pymongo import MongoClient
from tqdm import tqdm  # Import tqdm for progress bar

from agent import Agent
from typing import List, Dict, Tuple, OrderedDict

from learnPhysicsMapping import getInternalPhysics, process_state
from runBasePolicy import TD3, visualize, visualize2, writeMovie


def plotReward(reward_history):
  plt.plot(reward_history, label="Reward per Step")
  plt.xlabel("Steps")
  plt.ylabel("Reward")
  plt.title("Reward History over Episode")
  plt.legend()
  plt.show()

def getRewardTilde(qpos,qvel,action_sim):
  env_sim = suite.load(domain_name=domainName, task_name=taskName)
  env_sim.reset()
  sim_frames = []
  ret_reward = 0.
  with env_sim.physics.reset_context():
    env_sim.physics.data.qpos[:] = qpos
    env_sim.physics.data.qvel[:] = qvel
    state_hat = env_sim.step(action_sim)
    frame_sim = visualize(env_sim)
    sim_frames.append(frame_sim)
    reward_tilde_sim = state_hat.reward
    ret_reward += reward_tilde_sim
  env_sim.close()
  return ret_reward, sim_frames

if __name__ == '__main__':
  frames = []
  frames_tilde = []
  reward_history = []
  reward_tilde_history = []

  domainName = "walker"
  taskName = "stand"
  env_name = domainName+ "_"+taskName
  base_seed = 0 # Random seed number
  env = suite.load(domain_name=domainName, task_name=taskName)
  obsSpec = env.observation_spec()
  action_spec = env.action_spec()

  orientDim = obsSpec['orientations'].shape[0]
  heightDim = len(obsSpec['height'].shape) + 1 
  velocityDim = obsSpec['velocity'].shape[0]
  input_dim = orientDim + heightDim + velocityDim
  output_dim = env.action_spec().shape[0]


  state_dim = input_dim
  action_dim = output_dim
  max_action = float(1)
  base_policy = TD3(state_dim, action_dim, max_action)
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  file_name = "%s_%s_%s" % ("TD3", env_name, str(base_seed))
  base_policy.load(file_name, directory="./pytorch_models_backup")

  physicsModel = "/Users/athmajanvivekananthan/WCE/JEPA - MARL/multi-agent/dmcontrol/dmWalker/artifacts/Physics/510_walkerPhysics.pt"




  state = env.reset()
  obs = process_state(state)
  qpos,qvel = getInternalPhysics(obs,physicsModel)
  done = False
  epi_reward = 0.
  frame = visualize(env)
  frames.append(frame)
  while not done:
    action = base_policy.select_action(np.array(obs))
    state = env.step(action)
    
    obs = process_state(state)

    # qpos_,qvel_ = getInternalPhysics(obs,physicsModel)
    qpos = env.physics.data.qpos.copy()
    qvel = env.physics.data.qvel.copy()
    # feeding the same q values seem to produce same results. 
    # using validateDunamics_dynaPhysics.py to use NN with q state predictions
    import ipdb; ipdb.set_trace()



    reward_tilde, sim_frames1 = getRewardTilde(qpos,qvel,action)

    reward = state.reward
    done = state.last()
    epi_reward += reward
    reward_history.append(reward)
    reward_tilde_history.append(reward_tilde)
    frame = visualize(env)
    frames.append(frame)
    frames_tilde.append(sim_frames1[0])


  print(f"Reward is : {epi_reward}")
  writeMovie(frames,"validateDynamics")
  writeMovie(frames_tilde,"validateDynamics_tilde")
  plotReward(reward_history)
  plotReward(reward_tilde_history)

  # Plot reward history





    