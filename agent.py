import torch
import torch.nn as nn
import torch.optim as optim

from time import sleep

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

class Agent:
    def __init__(self, hyper):
        self.hyper = hyper

        self.policy_network = PolicyNetwork(input_size=1, output_size=2)
        self.target_network = PolicyNetwork(input_size=1, output_size=2)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.01)

    def select_action(self):
        return torch.randint(0, 2, (1,)).item()
    
    def optimize_model(self, memory):
        if len(memory) < 4:
            return
        transitions = memory.sample(4)

        state_batch = torch.tensor([t.state for t in transitions], 
                                   dtype=torch.float32).unsqueeze(1)
        action_batch = torch.tensor([t.action for t in transitions], 
                                    dtype=torch.int64).unsqueeze(1)
        next_batch = torch.tensor([t.next_state for t in transitions], 
                                    dtype=torch.float32).unsqueeze(1)
        reward_batch = torch.tensor([t.reward for t in transitions], 
                                    dtype=torch.float32).unsqueeze(1)
        
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(4, device=self.hyper.device)
        with torch.no_grad():
            next_state_values = self.target_network(next_batch).max(1)[0]
        expected_state_action_values = (next_state_values * self.hyper.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()

    def update_target_network(self, tau):
        pass