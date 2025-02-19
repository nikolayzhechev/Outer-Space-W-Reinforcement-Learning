import torch
import torch.nn as nn
import torch.optim as optim

# Custom Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)  # Batch normalization
        self.fc2 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.1)  # Dropout layer
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, action_dim)
        
    def forward(self, x):
        # Convert 1D tensor to 2D (batch_size=1)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure batch dimension exists
        x = torch.relu(self.bn1(self.fc1(x))) if x.shape[0] > 1 else torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)