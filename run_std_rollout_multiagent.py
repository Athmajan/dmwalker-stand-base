from agent_std_rollout import StdRolloutMultiAgent

import time
from typing import List
import cv2
import warnings
import logging

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from dm_control import suite
import torch
import json

from collections import deque, OrderedDict

N_EPISODES = 1
N_SIMS_PER_MC = 5
M_AGENTS = 2
NUM_BINS = 10
'''
By using the agent std rollout 
run the rollout
'''


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

def visualize2(env):
    frameA = np.hstack([env.physics.render(480, 480, camera_id=0),
                        env.physics.render(480, 480, camera_id=1)])
    plt.imshow(frameA)
    plt.pause(0.01)  # Need min display time > 0.0.
    plt.axis('off') 
    plt.draw()      
    plt.close() 
    return

def writeMovie(frames):
    # Define the codec and create a VideoWriter object
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('walker_stand_2.mp4', fourcc, 20.0, (width, height))

    # Write the frames to the video file
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    # Release the video writer
    out.release()
    return 



if __name__ == '__main__':
    np.random.seed(42)
    frames = []

    domainName = "walker" # Name of a environment (set it to any Continous environment you want)
    taskName = "stand" # Name of a environment (set it to any Continous environment you want)
    env_name = domainName+ "_"+taskName
    seed = 0 # Random seed number
    file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = suite.load(domain_name=domainName, task_name=taskName, task_kwargs={'random': seed})



    for i_episode in tqdm(range(N_EPISODES)):
        print("Running Episode", i_episode)

        state = env.reset()
        obs = process_state(state)

        std_rollout_multiagent = StdRolloutMultiAgent(
                M_AGENTS, NUM_BINS,env,device)
        

        done = False
        total_reward = .0
        ccc = 0
        while not done:
            act_n = std_rollout_multiagent.act_n(obs)
            print("found best action: ", act_n)
            
            state = env.step(act_n)
            frame = visualize(env)
            #visualize2(env)

            frames.append(frame)

            obs = process_state(state)
            reward = state.reward
            done = state.last()
            total_reward += np.sum(reward)

            dicttemp = {
                "Frame" : frame,
                "Best Action" :  act_n,
                "Observation" : obs, 
            }
            fileNameJson = "Episode_" + str(i_episode) + "_" + str(ccc) + ".json"

            with open(fileNameJson, 'w') as json_file:
                json.dump(dicttemp, json_file, indent=4)

            ccc= ccc +1

            

        print(f'Episode {i_episode}: Avg Reward is {total_reward / M_AGENTS}')


    env.close()
    writeMovie(frames)

        

        

