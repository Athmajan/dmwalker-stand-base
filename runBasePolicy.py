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


'''
evaluate the base policy using the pretrained 
DDPG model and record a movie
'''

domainName = "walker" # Name of a environment (set it to any Continous environment you want)
taskName = "stand" # Name of a environment (set it to any Continous environment you want)
env_name = domainName+ "_"+taskName
seed = 0 # Random seed number
batch_size = 100 # Size of the batch
discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005 # Target network update rate
policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated
     
file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")


# Discretizing Action Space
discretize = False
num_bins = 20
bin_edges = np.linspace(-1, 1, num_bins + 1)

# /Users/athmajanvivekananthan/WCE/JEPA - MARL/multi-agent/dmcontrol/dmWalker/artifacts/990_walkerDynamics.pt


class Network_Dynamics(nn.Module):
  def __init__(self):
    super(Network_Dynamics, self).__init__()
    self.linear1 = nn.Linear(30, 100)
    self.bn1 = nn.BatchNorm1d(100)
    self.linear2 = nn.Linear(100, 200)
    self.bn2 = nn.BatchNorm1d(200)
    self.linear3 = nn.Linear(200, 100)
    self.bn3 = nn.BatchNorm1d(100)
    self.linear4 = nn.Linear(100, 24)
    self.relu = nn.ReLU()


  def forward(self, x):
    x = self.linear1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.linear2(x)
    x = self.bn2(x)
    x = self.relu(x)
    x = self.linear3(x)
    x = self.bn3(x)
    x = self.relu(x)
    x = self.linear4(x)
    return x
  


class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
    for i in ind: 
      state, next_state, action, reward, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)
  

class Actor(nn.Module):
  
  def __init__(self, state_dim, action_dim, max_action):
    super(Actor, self).__init__()
    self.layer_1 = nn.Linear(state_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, action_dim)
    self.max_action = max_action

  def forward(self, x):
    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    x = self.max_action * torch.tanh(self.layer_3(x))
    return x
  
class Critic(nn.Module):
  
  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    # Defining the first Critic neural network
    self.layer_1 = nn.Linear(state_dim + action_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, 1)
    # Defining the second Critic neural network
    self.layer_4 = nn.Linear(state_dim + action_dim, 400)
    self.layer_5 = nn.Linear(400, 300)
    self.layer_6 = nn.Linear(300, 1)

  def forward(self, x, u):
    xu = torch.cat([x, u], 1)
    # Forward-Propagation on the first Critic Neural Network
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    # Forward-Propagation on the second Critic Neural Network
    x2 = F.relu(self.layer_4(xu))
    x2 = F.relu(self.layer_5(x2))
    x2 = self.layer_6(x2)
    return x1, x2

  def Q1(self, x, u):
    xu = torch.cat([x, u], 1)
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    return x1



class TD3(object):
  
  def __init__(self, state_dim, action_dim, max_action):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
    self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic = Critic(state_dim, action_dim).to(self.device)
    self.critic_target = Critic(state_dim, action_dim).to(self.device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    self.max_action = max_action

  def select_action(self, state):
    #print(state)
    state = torch.Tensor(state.reshape(1, -1)).to(self.device)
    return self.actor(state).cpu().data.numpy().flatten()

  def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    
    for it in range(iterations):
      
      # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      state = torch.Tensor(batch_states).to(self.device)
      next_state = torch.Tensor(batch_next_states).to(self.device)
      action = torch.Tensor(batch_actions).to(self.device)
      reward = torch.Tensor(batch_rewards).to(self.device)
      done = torch.Tensor(batch_dones).to(self.device)
      
      # Step 5: From the next state s’, the Actor target plays the next action a’
      next_action = self.actor_target(next_state)
      
      # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(self.device)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
      
      # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      target_Q1, target_Q2 = self.critic_target(next_state, next_action)
      
      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(state, action)
      
      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      
      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
  
  # Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))



   
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



# Function to discretize actions
def discretize_action(action, bin_edges):
    discretized_action = np.digitize(action, bin_edges) - 1
    discretized_action = np.clip(discretized_action, 0, len(bin_edges) - 2)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers[discretized_action]

# Function to query actions
def query_action(policy,observation, bin_edges):
    observation = process_state(observation)
    action = policy.select_action(np.array(observation))
    discretized_action = discretize_action(action, bin_edges)
    return discretized_action

def visualize2(env):
    frameA = np.hstack([env.physics.render(480, 480, camera_id=0),
                        env.physics.render(480, 480, camera_id=1)])
    plt.imshow(frameA)
    plt.axis('off')
    plt.show(block=False)  # Non-blocking show
    plt.pause(1)  # Display for 1 second
    plt.close()  # Close the figure after pause
    return


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



def queryDDPG(policy,obs,discretize=True,num_bins=20):
    if discretize:
       bin_edges = np.linspace(-1, 1, num_bins + 1)
       action = query_action(policy,obs, bin_edges)
    
    else:
       action = policy.select_action(np.array(obs))
  
    return action


def getRewardToCompare(savedSt, savedAct):
    env_sim = suite.load(domain_name=domainName, task_name=taskName, task_kwargs={'random': seed})
    env_sim.reset()
    with env_sim.physics.reset_context():
      env_sim.physics.data.qpos[:] = savedSt
      state_hat = env_sim.step(savedAct)
      reward_hat = state_hat.reward

    env_sim.close()
    return reward_hat

def evaluate_policy(policy, eval_episodes):
  # clf = Network_Dynamics().to(device)
  # clf.eval()
  # clf.load_state_dict(torch.load('/Users/athmajanvivekananthan/WCE/JEPA - MARL/multi-agent/dmcontrol/dmWalker/artifacts/990_walkerDynamics.pt', map_location=device))


  frames = []
  avg_reward = 0.
  ii= 0
  for _ in range(eval_episodes):
    epi_reward = 0
    epi_reward_hat = 0
    state = env.reset()
    obs = process_state(state)

    done = False
    while not done:
      if discretize:
        action = query_action(policy,obs, bin_edges) # discretized
      else:
         action = policy.select_action(np.array(obs))
      # import ipdb; ipdb.set_trace()
      state = env.step(action)
      # visualize2(env)
      # import ipdb; ipdb.set_trace()
      # savedState = env.physics.data.qpos.copy()  

      # calculate new state using NN here
      input_XU = np.concatenate((obs,action))
      # newState_Dynamic  = clf(torch.Tensor(input_XU.reshape(1,-1)))
      # newAction_Dynamic  = policy.select_action(newState_Dynamic.detach().numpy())

      # Now restart the envirnoment from the previous state
      # and apply this new action to get the reward.
      # reward_hat = getRewardToCompare(savedState,newAction_Dynamic)


      frame = visualize(env)
      frames.append(frame)
      obs = process_state(state)
      reward = state.reward
      done = state.last()
      avg_reward += reward
      epi_reward += reward
      print(epi_reward)
      # epi_reward_hat += reward_hat

         

    if logginWB:
      wandb.log({'episode_reward':epi_reward,
                #  'episode_reward_hat':epi_reward_hat,
                    },step=ii)
    ii += 1 
      

  avg_reward /= eval_episodes
  print ("---------------------------------------")
  print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
  print ("---------------------------------------")
  return avg_reward, frames



if __name__ == '__main__':
  logginWB = False
  if logginWB:
    import wandb
    wandb.init(project="dmWalker",name=f"runBasePolicy_w_rewardHat",save_code=True)
    
  eval_episodes = 1
  save_env_vid = True
  env = suite.load(domain_name=domainName, task_name=taskName, task_kwargs={'random': seed})
  torch.manual_seed(seed)
  np.random.seed(seed)
  random_state = np.random.RandomState(seed)


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

  # Selecting the device (CPU or GPU)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))


  policy = TD3(state_dim, action_dim, max_action)

  policy.load(file_name, directory="./pytorch_models_backup")

  # Number of bins (hyperparameter)

  avgReward, frames = evaluate_policy(policy, eval_episodes=eval_episodes)
  writeMovie(frames,'testBP')