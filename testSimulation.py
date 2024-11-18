from agent import MultiAgent
from typing import List, Dict, Tuple
from agent_ddpg import DDPGAgent, TD3
import torch
from dm_control import suite
import numpy as np

from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from operator import itemgetter

from collections import deque, OrderedDict


   
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



            # initial_obs: List[float],
            # initial_step: Tuple, # config in act_n -> initial actions
            # m_agents: int,
            # num_bins: int,
            # n_sim_per_step: int,
            # env,
        



def simulate(
            initial_obs: List[float],
            initial_step: Tuple, # config in act_n -> initial actions
            m_agents: int,
            num_bins: int,
            n_sim_per_step: int,
            env,
    ) -> Tuple[Tuple, float]:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obsSpec = env.observation_spec()

        orientDim = obsSpec['orientations'].shape[0]
        heightDim = len(obsSpec['height'].shape) + 1 
        velocityDim = obsSpec['velocity'].shape[0]
        input_dim = orientDim + heightDim + velocityDim

        output_dim = env.action_spec().shape[0]


        state_dim = input_dim
        action_dim = output_dim
        max_action = float(1)
        policy = TD3(state_dim, action_dim, max_action, device)

        ## Here the agents are created in the order 1 2 3 4 ...
        ## However an improvement can be done by checking Q values for
        ## all agents and reordering
        # create agents

        agents = [
            DDPGAgent(
                i, num_bins,policy, device
            ) for i in range(m_agents)
        ]

        # run N simulations
        avg_total_reward = .0
        for _ in range(n_sim_per_step):
            with env.physics.reset_context():
                env.physics.data.qpos[:] = initial_obs

                # 1 step
                state = env.step(initial_step)
                reward = state.reward
            
                obs = process_state(state)




                #avg_total_reward += np.sum(reward)

                # Run an episode until end
                done = False
                while not done:
                    act_n = []
                    for agent in agents:
                        indAct , totAct = agent.act(obs)
                        act_n.append(indAct)

                    act_n = np.concatenate(act_n)
                    state = env.step(act_n)
                    reward = state.reward
                    done = state.last()

                    avg_total_reward += np.sum(reward)

        env.close()

        avg_total_reward /= m_agents
        avg_total_reward /= n_sim_per_step

        return initial_step, avg_total_reward




seed = 0 # Random seed number
np.random.seed(seed)
# Create the HalfCheetah environment
env = suite.load(domain_name="walker", task_name="stand", task_kwargs={'random': seed})
# Reset the environment to get the initial state
state = env.reset()
action_spec = env.action_spec()

time_step = env.reset()

savedFrame = {}
savedAct = {}
i = 0
while not time_step.last():
    action = np.random.uniform(action_spec.minimum,
                            action_spec.maximum,
                            size=action_spec.shape)
    
    time_step = env.step(action)
    
    i += 1
    if i == 200:
        savedFrame[i] = env.physics.data.qpos.copy()  
        savedAct[i] = env.physics.data.ctrl.copy()  

    elif i == 600:
        savedFrame[i] = env.physics.data.qpos.copy()    
        savedAct[i] = env.physics.data.ctrl.copy()  
    elif i == 1000:
        savedFrame[i] = env.physics.data.qpos.copy() 
        savedAct[i] = env.physics.data.ctrl.copy()       
    elif i == 800:
        savedFrame[i] = env.physics.data.qpos.copy() 
        savedAct[i] = env.physics.data.ctrl.copy()  
    


print(savedAct[200])

initial_step, avg_total_reward = simulate(savedFrame[200],savedAct[200],2,20,10,env)
print(initial_step)
print(avg_total_reward)