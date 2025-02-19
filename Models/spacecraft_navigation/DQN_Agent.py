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

# Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.4): # tested wotj 0.6
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
    
    def add(self, experience, priority=1.0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            # Replace the oldest experience
            self.buffer.pop(0)
            self.priorities.pop(0)
            self.buffer.append(experience)
            self.priorities.append(priority)
    
    def sample(self, batch_size):
        scaled_priorities = np.array(self.priorities) ** self.alpha
        probs = scaled_priorities / scaled_priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        return samples, indices
    
    def update_priorities(self, indices, errors, epsilon=1e-6):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + epsilon

# DQN Agent with Improvements
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = PrioritizedReplayBuffer(capacity=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.batch_size = 64
        self.learning_rate = 0.0003
        
        self.model = DQN(state_dim, action_dim).float()
        self.target_model = DQN(state_dim, action_dim).float()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.9)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()
    
    def replay(self):
        if len(self.memory.buffer) < self.batch_size:
            return
        minibatch, indices = self.memory.sample(self.batch_size)
        
        errors = []
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            
            # Compute target Q-values
            target = self.model(state_tensor).detach()
            if done:
                target[0, action] = reward_tensor
            else:
                with torch.no_grad():
                    t = self.target_model(next_state_tensor)
                target[0, action] = reward_tensor + self.gamma * torch.max(t)
            
            # Get current prediction and compute loss
            output = self.model(state_tensor)
            loss = nn.functional.mse_loss(output, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            
            errors.append(loss.item())
        
        self.scheduler.step()
        self.memory.update_priorities(indices, errors)
        # Adaptive epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * (0.995 if self.epsilon > 0.1 else 0.999))
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())