import torch
from singleagent.agent import SingleAgent
from singleagent.train import train_agent
from singleagent.memory import ReplayMemory

class RLPlug(object):
    def __init__(self, environment, params):
        self.environment = environment
        self.params = params
        self.agent = SingleAgent(environment, params)
        self.memory = ReplayMemory(1000)

    def train(self):
        train_agent(self.agent, self.environment, self.params, self.memory)