from environment import Environment
from agent import Agent
from train import train_agents

if __name__ == "__main__":
    # Initialize environment and agents
    environment = Environment(num_agents=4)
    agents = [Agent(environment) for _ in range(environment.num_agents)]
    
    # Train agents
    train_agents(agents, environment)
