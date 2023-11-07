import torch

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