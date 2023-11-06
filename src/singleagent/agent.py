import torch
import torch.nn as nn
import torch.optim as optim
import random
import math

from time import sleep

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.input_layer = nn.Linear(input_size, 10)
        self.output_layer = nn.Linear(10, output_size)

    def forward(self, x):
        x = nn.functional.relu(self.input_layer(x))
        return self.output_layer(x)

class SingleAgent:
    def __init__(self, environment, hyper):
        self.hyper = hyper
        self.environment = environment

        state_dim = len(environment.state)
        n_actions = len(environment.actions)

        self.policy_network = PolicyNetwork(input_size=state_dim, output_size=n_actions).to(hyper.device)
        self.target_network = PolicyNetwork(input_size=state_dim, output_size=n_actions).to(hyper.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=hyper.learning_rate)
        self.steps_done = 0

    def select_action(self, state):
        n = random.random()
        eps_threshold = self.hyper.eps_end + (self.hyper.eps_start - self.hyper.eps_end) * \
            math.exp(-1 * self.steps_done / self.hyper.eps_decay)
        self.steps_done += 1
        if n < eps_threshold:
            res = torch.randint(0, 2, (1,))
            return res
        else:
            with torch.no_grad():
                res = self.policy_network(state).max(1)[1].view(1,1)
                return res
    
    def optimize_model(self, memory):
        if len(memory) < self.hyper.batch_size:
            return
        transitions = memory.sample(self.hyper.batch_size)

        state_batch = torch.tensor([t.state for t in transitions], 
                                   dtype=torch.float32, device=self.hyper.device).unsqueeze(1)
        action_batch = torch.tensor([t.action for t in transitions], 
                                    dtype=torch.int64, device=self.hyper.device).unsqueeze(1)
        next_batch = torch.tensor([t.next_state for t in transitions], 
                                    dtype=torch.float32, device=self.hyper.device).unsqueeze(1)
        reward_batch = torch.tensor([t.reward for t in transitions], 
                                    dtype=torch.float32, device=self.hyper.device).unsqueeze(1)
        
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(self.hyper.batch_size, device=self.hyper.device)
        with torch.no_grad():
            next_state_values = self.target_network(next_batch).max(1)[0].unsqueeze(1)
        expected_state_action_values = (next_state_values * self.hyper.gamma) + reward_batch
        
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()

    def update_target_network(self):
        target_net_state_dict = self.target_network.state_dict()
        policy_net_state_dict = self.policy_network.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.hyper.tau \
                + target_net_state_dict[key] * (1-self.hyper.tau)
        self.target_network.load_state_dict(target_net_state_dict)