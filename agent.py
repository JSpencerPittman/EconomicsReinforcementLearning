import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

class Agent:
    def __init__(self, environment):
        self.policy_network = PolicyNetwork(input_size=1, output_size=2)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.01)
        self.env = environment

    def select_action(self, state):
        return torch.randint(0, 2, (1,)).item()
