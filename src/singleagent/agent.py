import torch
import torch.nn as nn
import torch.optim as optim
import random
import math

from singleagent.memory import ReplayMemory
from singleagent.params import HyperParameters
from singleagent.environment import Environment

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.input_layer = nn.Linear(input_size, 10)
        self.output_layer = nn.Linear(10, output_size)

    def forward(self, x):
        x = nn.functional.relu(self.input_layer(x))
        return self.output_layer(x)

class SingleAgent:
    def __init__(self, environment: Environment, hyper: HyperParameters):
        self.hyper = hyper
        self.environment = environment

        self.state_dim = len(environment.state)
        self.n_actions = len(environment.actions)

        self.policy_network = PolicyNetwork(input_size=self.state_dim, output_size=self.n_actions).to(hyper.device)
        self.target_network = PolicyNetwork(input_size=self.state_dim, output_size=self.n_actions).to(hyper.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=hyper.learning_rate)
        
        self.steps_done = 0

        self.explore_actions = [0] * self.n_actions
        self.exploit_actions = [0] * self.n_actions

    def select_action(self, state: torch.tensor):
        """
        Selects an action for the agent to take using the Epsilon-Greedy Algorithm

        state: current state of the system
        """

        x = random.random()
        eps_threshold = self.hyper.eps_end + (self.hyper.eps_start - self.hyper.eps_end) * \
            math.exp(-1 * self.steps_done / self.hyper.eps_decay)
        
        self.steps_done += 1
        
        # Explore
        if x < eps_threshold:
            res = torch.randint(0, self.n_actions, (1,))
            self.explore_actions[res] += 1
            return res
        # Exploit
        else:
            with torch.no_grad():
                # Choose the action that has the highest Q-value
                res = self.policy_network(state).max(1)[1].view(1,1)
                self.exploit_actions[res] += 1
                return res
    
    def optimize_policy(self, memory: ReplayMemory):
        """
        Update the policy network

        memory: memory of previous iterations
        """

        # Make sure enough trials have been saved
        if len(memory) < self.hyper.batch_size:
            return
        
        # Grab BATCH_SIZE transitions from the memory
        transitions = memory.sample(self.hyper.batch_size)

        # Split the transitions into states, actions, next_states, and rewards
        state_batch = torch.tensor([t.state for t in transitions], 
                                   dtype=torch.float32, device=self.hyper.device).unsqueeze(1)
        action_batch = torch.tensor([t.action for t in transitions], 
                                    dtype=torch.int64, device=self.hyper.device).unsqueeze(1)
        next_batch = torch.tensor([t.next_state for t in transitions], 
                                    dtype=torch.float32, device=self.hyper.device).unsqueeze(1)
        reward_batch = torch.tensor([t.reward for t in transitions], 
                                    dtype=torch.float32, device=self.hyper.device).unsqueeze(1)
        
        # Calculate the Q-Values for the previously taken actions
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)
        
        # Find the max Q-Value for each next_state
        with torch.no_grad():
            next_state_values = self.target_network(next_batch).max(1)[0].unsqueeze(1)
        # Reward is equal to the current reward + (discount factor * next state's reward)
        expected_state_action_values = (next_state_values * self.hyper.gamma) + reward_batch
        
        # Calculate total loss 
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Reset gradients to zero because pytorch accumalates this automatically
        self.optimizer.zero_grad()
        # Perform backpropagation to calculate gradients
        loss.backward()
        # Ensure the gradients don't exceed 100 to prevent exploding gradients
        torch.nn.utils.clip_grad.clip_grad_value_(self.policy_network.parameters(), 100)
        # Update policy_networks weights using calculated gradients
        self.optimizer.step()

    def optimize_target(self):
        # Get current weights
        target_net_state_dict = self.target_network.state_dict()
        policy_net_state_dict = self.policy_network.state_dict()

        # Align the target network a little more with the policy network
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.hyper.tau \
                + target_net_state_dict[key] * (1-self.hyper.tau)
            
        # Save the shifted weights to the target network
        self.target_network.load_state_dict(target_net_state_dict)