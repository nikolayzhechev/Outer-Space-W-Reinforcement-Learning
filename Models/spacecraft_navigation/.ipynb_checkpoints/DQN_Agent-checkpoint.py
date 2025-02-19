import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from collections import deque

import sys
import os
# Parent directory is in the path
sys.path.append(os.path.abspath("../Models/spacecraft_navigation"))
# Import modules
from DQN_NN import DQN

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.batch_size = 64
        # self.learning_rate = 0.0003  # Reduced learning rate
        self.learning_rate = 5e-4
        
        self.model = DQN(state_dim, action_dim).float()
        self.target_model = DQN(state_dim, action_dim).float()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.9)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            reward = torch.tensor(reward, dtype=torch.float32)
            
            target = self.model(state).detach()
            if done:
                target[0, action] = reward
            else:
                with torch.no_grad():
                    t = self.target_model(next_state)
                if 0 <= action < self.action_dim:
                    target[0, action] = reward + self.gamma * torch.max(t)
            
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            self.optimizer.step()
        
        self.scheduler.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * (0.995 if self.epsilon > 0.1 else 0.999))
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())