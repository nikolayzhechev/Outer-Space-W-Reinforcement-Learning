import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.distributions import Normal

# Dueling DQN Network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Separate streams for value and advantage
        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, action_dim)
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure batch dimension exists
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # Combine value and advantage streams
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q