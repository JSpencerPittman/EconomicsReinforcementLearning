from abc import ABC, abstractmethod
from enum import Enum
from collections import deque
import random
import torch

class Environment(ABC):
    def __init__(self, actions: Enum):
        self.actions = actions
        self.state = list()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

class HyperParameters(object):
    def __init__(self, 
                 episodes=100,
                 steps=100, 
                 gamma=0.99,
                 tau=0.005,
                 eps_start=0.9,
                 eps_end=0.05,
                 eps_decay=1000,
                 learning_rate=1e-4,
                 batch_size=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.episodes = episodes
        self.steps = steps
        self.gamma = gamma
        self.tau = tau
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size

class Plotter(object):
    def __init__(self, plot_function, frequency=30):
        self.plot_function = plot_function
        self.frequency = frequency

    def plot(self, states, exploits, explores, policy_net, hyper, i_episode, show_result=False):
        if i_episode % self.frequency != 0:
            return
        
        self.plot_function(states, 
                            exploits, 
                            explores, 
                            policy_net, 
                            hyper, 
                            i_episode, 
                            show_result)
        
class Transition(object):
    def __init__(self, state, action, next_state, reward):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], capacity)

    def push(self, transition):
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)