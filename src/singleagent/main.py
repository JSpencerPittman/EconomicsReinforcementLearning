from example.environment import Environment
from singleagent.agent import Agent
from singleagent.train import train_agent, HyperParameters
from singleagent.memory import ReplayMemory

if __name__ == "__main__":
    # Initialize environment and agents
    params = HyperParameters(episodes=300, batch_size=16, eps_decay=4000, gamma=0.9, learning_rate=1e-3)

    environment = Environment()
    agent = Agent(params)
    memory = ReplayMemory(1000)
    
    # Train agents
    res = train_agent(agent, environment, params, memory)