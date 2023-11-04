import torch

from agent import Agent
from environment import Environment
from memory import ReplayMemory, Transition

class HyperParameters(object):
    def __init__(self, 
                 episodes=100, 
                 gamma=0.99,
                 tau=0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.episodes = episodes
        self.gamma = gamma
        self.tau = tau

def train_agent(agent: Agent, environment: Environment, hyper: HyperParameters, memory: ReplayMemory):
    results = list()

    for i_episode in range(hyper.episodes):  # Number of episodes
        state = environment.reset()
        state = torch.tensor(state, dtype=torch.float32, device=hyper.device).unsqueeze(0)

        for t in range(10):
            action = agent.select_action(state)
            next_state, rewards = environment.step(action)
            rewards = torch.tensor(rewards, device=hyper.device)
            
            next_state = torch.tensor(next_state, dtype=torch.float32, device=hyper.device).unsqueeze(0)
            
            trans = Transition(state, action, next_state, rewards)
            memory.push(trans)

            state = next_state

            agent.optimize_model(memory)
            agent.update_target_network()

        results.append(state)

    return results

    