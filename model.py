import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from torch.utils.data import DataLoader


# 4. MLP Model
class SimpleMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # (batch_size, 1, 28, 28) --> (batch_size, 784)
        x = self.fc1(x)      # (batch_size, 784) --> (batch_size, hidden_size)
        x = self.relu(x)     # (batch_size, hidden_size) --> (batch_size, hidden_size)
        x = self.fc2(x)      # (batch_size, hidden_size) --> (batch_size, 32)
        x = self.relu(x)     # (batch_size, 32) --> (batch_size, 32)
        logits = self.fc3(x) # (batch_size, 32) --> (batch_size, 10)

        return logits