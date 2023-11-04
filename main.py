from environment import Environment
from agent import Agent
from train import train_agent, HyperParameters
from memory import ReplayMemory

if __name__ == "__main__":
    # Initialize environment and agents
    params = HyperParameters(episodes=100)

    environment = Environment()
    agent = Agent(params)
    memory = ReplayMemory(10)
    
    # Train agents
    res = train_agent(agent, environment, params, memory)

    res = [v.item() for v in res]

    print(res)
