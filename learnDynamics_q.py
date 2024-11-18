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

from agent import Agent
from typing import List, Dict, Tuple, OrderedDict
from tqdm import tqdm 

'''
Train a NN to approximate the state dynamics.
The purpose is to use this for MPC.
'''

domainName = "walker" # Name of a environment (set it to any Continous environment you want)
taskName = "stand" # Name of a environment (set it to any Continous environment you want)
env_name = domainName+ "_"+taskName

def insertToMongo(data):
    client = MongoClient("mongodb://localhost:27017/")  # Adjust the connection string if needed
    # Select the database and collection
    db = client['dmwalker']  # Replace 'your_database_name' with your actual database name
    collection = db[f'dmwalker_dynaPhysics']  # Replace 'your_collection_name' with your actual collection name
    collection.insert_many(data)
    return

    


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


def retrieveRandomFromMongo(num_entries):
    client = MongoClient("mongodb://localhost:27017/")
    db = client['dmwalker']
    collection = db['dmwalker_dynaPhysics']
    pipeline = [
        {"$sample": {"size": num_entries}}
    ]
    all_documents = list(collection.aggregate(pipeline))

    outputSamples = []
    for doc in all_documents:
        x = np.array(doc["state"], dtype=np.float32)
        u = np.array(doc["action"], dtype=np.float32)
        xu = np.concatenate((x, u))
        y = np.array(doc["next_state"], dtype=np.float32)
        outputSamples.append((xu,y))

    return outputSamples

def gatherData(booStrapN = 1000000):
    batch_size = 1000
    batch_data = []
    env = suite.load(domain_name=domainName, task_name=taskName)

    obsSpec = env.observation_spec()
    action_spec = env.action_spec()


    orientDim = obsSpec['orientations'].shape[0]
    heightDim = len(obsSpec['height'].shape) + 1 
    velocityDim = obsSpec['velocity'].shape[0]
    input_dim = orientDim + heightDim + velocityDim
    output_dim = env.action_spec().shape[0]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = env.reset()
    qPos = np.array(env.physics.data.qpos.copy(), dtype=np.float32)
    qVel = np.array(env.physics.data.qvel.copy(), dtype=np.float32)
    qLabel = np.concatenate((qPos, qVel))
    for _ in tqdm(range(booStrapN), desc="Gathering Data"):
        randAction =  np.random.uniform(min(env.action_spec().minimum),
                                        max(env.action_spec().maximum), 
                                        (output_dim,))
        done = state.last()
        if done:
            state = env.reset()
            qPos = np.array(env.physics.data.qpos.copy(), dtype=np.float32)
            qVel = np.array(env.physics.data.qvel.copy(), dtype=np.float32)


        obs_prev = process_state(state)

        qPos = np.array(env.physics.data.qpos.copy(), dtype=np.float32)
        qVel = np.array(env.physics.data.qvel.copy(), dtype=np.float32)
        qLabel_prev = np.concatenate((qPos, qVel))

        state = env.step(randAction)

        qPos = np.array(env.physics.data.qpos.copy(), dtype=np.float32)
        qVel = np.array(env.physics.data.qvel.copy(), dtype=np.float32)
        qLabel_next = np.concatenate((qPos, qVel))

        new_obs = process_state(state)


        dataSample = {
            "state" : qLabel_prev.tolist(),
            "action" : randAction.tolist(),
            "next_state" : qLabel_next.tolist(),
                    }
        batch_data.append(dataSample)

        if len(batch_data) >= batch_size:
            insertToMongo(batch_data)
            batch_data = []


    if batch_data:
        insertToMongo(batch_data)
class Network(nn.Module):
  def __init__(self):
    super(Network, self).__init__()
    self.linear1 = nn.Linear(inputDim, hidden_layer_1)
    self.bn1 = nn.BatchNorm1d(hidden_layer_1)
    self.linear2 = nn.Linear(hidden_layer_1, hidden_layer_2)
    self.bn2 = nn.BatchNorm1d(hidden_layer_2)
    self.linear3 = nn.Linear(hidden_layer_2, hidden_layer_3)
    self.bn3 = nn.BatchNorm1d(hidden_layer_3)
    self.linear4 = nn.Linear(hidden_layer_3, outputDim)
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
  
def runTestEvaluation():
    clf.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    n_batches = 0

    with torch.no_grad():  # No need to calculate gradients during testing
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = clf(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            n_batches += 1

    avg_test_loss = test_loss / n_batches
    print(f"Test Loss: {avg_test_loss}")
    clf.train()  # Set the model back to training mode
    return avg_test_loss



if __name__ == '__main__':
    elementWiseThresh = 0.01
    margin = 0.01
    beta = 0.5
    gamma = 0.5

    WANDB = True
    if WANDB:
        import wandb
        wandb.init(project="dmWalker",name=f"dynaPhysics",save_code=True)
    inputDim  = 24
    outputDim = 18
    hidden_layer_1 = 100
    hidden_layer_2 = 200
    hidden_layer_3 = 100


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = Network().to(device)
    optimizer = optim.Adam(clf.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)
    criterion = nn.MSELoss()



    totalSamples = 400000
    batchSize = 2000
    trainEpochs = 1000

    dataSet = retrieveRandomFromMongo(totalSamples)

    train_size = int(0.8 * totalSamples)
    test_size = totalSamples - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataSet, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batchSize,
                                              shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize,shuffle=False)
    
    testLoss = runTestEvaluation()
    for epoch in range(trainEpochs):
        running_loss = .0
        n_batches = 0


        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = clf(inputs)

            mse_loss = F.mse_loss(outputs, labels)
            # mae_loss = F.l1_loss(outputs, labels)
            # total_loss = mse_loss + alpha * mae   _loss

            thresholds = torch.full_like(outputs, elementWiseThresh, device=device)  # define acceptable error thresholds
            errors = torch.abs(outputs - labels)
            threshold_violation = F.relu(errors - thresholds)  # penalizes only if error exceeds threshold
            total_loss = mse_loss + gamma * threshold_violation.mean()  # adjust gamma as needed

            # # allowable deviation
            # diff = torch.abs(outputs - labels) - margin
            # hinge_loss = torch.mean(F.relu(diff))  # penalizes only if outside margin
            # total_loss = mse_loss + beta * hinge_loss  # adjust beta for balance
            
            total_loss.backward()

            optimizer.step()
            running_loss += total_loss.item()
            n_batches += 1

        epochAvg_loss = running_loss / n_batches
        print(f"Epoch {epoch}, Loss = {epochAvg_loss}")

        if epoch % 10 == 0:
            testLoss = runTestEvaluation()
            torch.save(clf.state_dict(), f'artifacts/dynaPhysics2/{epoch}_walkerDynamics.pt')

        if WANDB:
            wandb.log({'training_loss':epochAvg_loss,
                    'test_loss' : testLoss,
                    'Learning Rate': optimizer.param_groups[0]['lr'],
                    },step=epoch) 
        scheduler.step(epochAvg_loss)


# /Users/athmajanvivekananthan/WCE/JEPA - MARL/multi-agent/dmcontrol/dmWalker/artifacts/dynaPhysics2/470_walkerDynamics.pt
    



