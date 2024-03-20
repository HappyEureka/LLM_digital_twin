import math
import random
#import matplotlib
#import matplotlib.pyplot as plt
from collections import namedtuple, deque
#from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, model_complexity=256):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, model_complexity)
        self.layer2 = nn.Linear(model_complexity, model_complexity)
        #self.layer3 = nn.Linear(4, 4)
        self.layer4 = nn.Linear(model_complexity, n_actions)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        #x = F.relu(self.layer3(x))
        return self.layer4(x)
    

class ActionSpace():
    def __init__(self):
        self.true_action_space = np.round(np.arange(17, 28, 0.5), 2)
        self.action_space = np.arange(0, len(self.true_action_space), 1) 
        self.n = len(self.action_space)

    def convert(self, n):
        return self.true_action_space[n]
    
    def sample(self):
        return round(self.action_space[np.random.randint(0, self.n)], 2)
    
#action_space = ActionSpace()
#n_actions = action_space.n



BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
TAU = 0.005
LR = 1e-4
class RL():
    def __init__(self, n_observations, n_actions, action_space, model_complexity=256, lr_=1e-4, batch_size_=128):
        global BATCH_SIZE, LR
        BATCH_SIZE = batch_size_
        LR = lr_

        self.n_actions = n_actions
        self.n_observations = n_observations
        self.memory = ReplayMemory(100000)
        self.policy_net = DQN(n_observations, n_actions, model_complexity).to(device)
        self.target_net = DQN(n_observations, n_actions, model_complexity).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        
        self.target_net_state_dict = None
        self.policy_net_state_dict = None
        self.state = None
        self.reward = None
        self.next_state = None
        
        self.steps_done = 0
        self.action_space = action_space

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.action_space.sample()]], device=device, dtype=torch.long)


    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def step(self, action, state, next_state, next_reward):
        self.reward = torch.tensor([next_reward], device=device)
        self.next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
        # Store the transition in memory
        self.memory.push(state, action, self.next_state, self.reward)
        # self.state = self.next_state
        self.optimize_model()
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)
        
        return self.next_state
    
    def predict(self, state):
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)