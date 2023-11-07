import torch
from singleagent.agent import SingleAgent
from singleagent.train import train_agent
from singleagent.memory import ReplayMemory
from singleagent.params import HyperParameters
from singleagent.environment import Environment
from singleagent.plot import Plotter

class RLPlug(object):
    def __init__(self, environment: Environment, hyper: HyperParameters, plotter: Plotter):
        self.environment = environment
        self.hyper = hyper
        self.plotter = plotter
        self.agent = SingleAgent(environment, hyper)
        self.memory = ReplayMemory(1000)

    def train(self):
        return train_agent(self.agent, self.environment, self.hyper, self.memory, self.plotter)