from abc import ABC, abstractmethod
from enum import Enum

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