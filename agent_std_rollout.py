from agent import MultiAgent
from typing import List, Dict, Tuple
from collections import deque, OrderedDict
from agent_ddpg import DDPGAgent, TD3
import torch
from dm_control import suite
import numpy as np

from itertools import product, chain
from concurrent.futures import ProcessPoolExecutor, as_completed
from operator import itemgetter
from tqdm import tqdm

from multiprocessing import Pool


class StdRolloutMultiAgent(MultiAgent):
    def __init__(
            self,
            m_agents: int,
            num_bins: int,
            env,
            device,
            n_sim_per_step: int = 2,
            n_workers: int = 10,
            
    ):
        self._m_agents = m_agents
        self._num_bins = num_bins
        self._n_sim_per_step = n_sim_per_step
        self._n_workers = n_workers
        self._env = env
        self._device = device

    def _chunk_configs(self, configs, chunk_size):
        """Divide configs into chunks."""
        for i in range(0, len(configs), chunk_size):
            yield configs[i:i + chunk_size]


    def act_n(
            self,
            obs: List[float],
            **kwargs,
    ) :
        bin_edges = np.linspace(-1, 1, self._num_bins + 1)
        

        available_moves_per_joint = (bin_edges[:-1] + bin_edges[1:]) / 2
        # each leg has three action choices (motors)
        available_moves_per_agent = list(product(available_moves_per_joint, repeat=3))

        obsSpec = self._env.observation_spec()

        orientDim = obsSpec['orientations'].shape[0]
        heightDim = len(obsSpec['height'].shape) + 1 
        velocityDim = obsSpec['velocity'].shape[0]
        state_dim = orientDim + heightDim + velocityDim
        action_dim = self._env.action_spec().shape[0]
        max_action = float(1)
        policy = TD3(state_dim, action_dim, max_action, self._device)



        agents = [
            DDPGAgent(
                i, self._num_bins,policy, self._device
            ) for i in range(self._m_agents)
        ]

        indAct = []
        for agent in agents:
            if agent.id ==1 :
                indAct, totAct = agent.act(obs)
                print(indAct, totAct)

        indActTuple = ()
        for act2nd in indAct:
            indActTuple = indActTuple + (float(act2nd),)


        configs = []
        for move in available_moves_per_agent:
            config = move + indActTuple
            configs.append(config)

        sim_results = []
        qPos = self._env.physics.data.qpos.copy()
        config_chunks = list(self._chunk_configs(configs, 100))

        with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
            futures = [
                executor.submit(self._simulate_chunk, qPos, chunk, self._m_agents, self._num_bins, self._n_sim_per_step, self._device)
                for chunk in config_chunks
            ]

            for f in tqdm(as_completed(futures), total=len(futures), desc="Simulating", unit="chunk"):
                res = f.result()
                sim_results.extend(res)

        best_config = max(sim_results, key=lambda x: x[1])[0]

        best_configAgent1 = best_config[:3]

        indActTuple2 = ()
        for act2nd in best_configAgent1:
            indActTuple2 = indActTuple2 + (float(act2nd),)

        configs = []
        for move in available_moves_per_agent:
            config = indActTuple2 + move
            configs.append(config)

        sim_results = []
        
        config_chunks = list(self._chunk_configs(configs, 100))

        with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
            futures = [
                executor.submit(self._simulate_chunk, qPos, chunk, self._m_agents, self._num_bins, self._n_sim_per_step, self._device)
                for chunk in config_chunks
            ]

            for f in tqdm(as_completed(futures), total=len(futures), desc="Simulating", unit="chunk"):
                res = f.result()
                sim_results.extend(res)

        best_config = max(sim_results, key=lambda x: x[1])[0]

        return best_config
            



    @staticmethod
    def _simulate_chunk(qPos, config_chunk, m_agents, num_bins, n_sim_per_step, device):
        results = []
        for config in config_chunk:
            #config = list(chain(*config))
            result = StdRolloutMultiAgent._simulate(qPos, config, m_agents, num_bins, n_sim_per_step, device)
            results.append(result)
        return results




       
    
    @staticmethod
    def _process_state(state):
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
            raise ValueError("InpPut state must be either an OrderedDict with keys 'orientations', 'height', and 'velocity', a numpy ndarray of shape (24,), or a TimeStep object with a valid observation.")




    @staticmethod
    def _simulate(
            init_qPos,
            initial_step: Tuple, # config in act_n -> initial actions
            m_agents: int,
            num_bins: int,
            n_sim_per_step: int,
            device,
    ) -> Tuple[Tuple, float]:
        
        env = suite.load(domain_name="walker", task_name="stand", task_kwargs={'random': 0})
        env.reset()
        
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
                env.physics.data.qpos[:] = init_qPos

                # 1 step
                state = env.step(initial_step)

                reward = state.reward
                obs = StdRolloutMultiAgent._process_state(state)

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
        #print(initial_step, avg_total_reward)
        return initial_step, avg_total_reward