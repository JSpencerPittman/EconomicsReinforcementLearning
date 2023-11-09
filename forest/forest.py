from singleagent.train import train_agent
from enum import Enum
from squaregrid import SquareGrid
from singleagent.agent import SingleAgent
from singleagent.utils import ( Environment,
                                HyperParameters,
                                ReplayMemory,
                                Plotter )
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from scipy.signal import savgol_filter

class Actions(Enum):
    IGNORE = 0
    WATER = 1

def grid_cross_summation(grid: SquareGrid, r, c):
    points = [(r,c),(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
    return sum([grid.val(*p) for p in points if grid.inbounds(*p)])

class Fire(Environment):
    def __init__(self, start_state: SquareGrid):
        super(Fire, self).__init__()
        self.state = start_state
        self.start_state = start_state
        self.side = start_state.shape()[0]
        self.n_actions = self.side ** 2
        self.state_dim = self.side ** 2

    def reset(self):
        self.state = self.start_state
        return self.state.ravel()

    def step(self, actions):
        next_state = self.state

        for r in range(self.side):
            for c in range(self.side):
                cell_val = grid_cross_summation(self.state, r, c)
                cell_val = max(cell_val, 3)
                next_state.update(r, c, cell_val)
        
        for i, action in enumerate(actions):
            if action == Actions.WATER.value:
                next_state.update(math.floor(i/self.side), i % self.side, 0)
        
        clear_tiles = 0
        for r in range(self.side):
            for c in range(self.side):
                if next_state.val(r,c) <= 0:
                    clear_tiles += 1

        self.state = next_state

        return self.state.ravel(), [clear_tiles]
            
def plot_progress(states, exploits, explores, policy_net, hyper, i_episode, show_result):
    print("PROGRESS")
    print(states)

start_state = SquareGrid([[0,0,0,1],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]])

params = HyperParameters(episodes=100, 
                         steps=10,
                         batch_size=16, 
                         eps_decay=4000, 
                         gamma=0.9, 
                         learning_rate=1e-3)

environment = Fire(start_state)
plotter = Plotter(plot_progress)
agent = SingleAgent(environment, params)
memory = ReplayMemory(1000)


print(train_agent(agent, environment, params, memory, plotter))
