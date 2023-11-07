import torch
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np

from singleagent.agent import SingleAgent
from singleagent.memory import ReplayMemory, Transition
from singleagent.environment import Environment
from singleagent.params import HyperParameters
from singleagent.plot import Plotter

def train_agent(agent: SingleAgent, environment: Environment, hyper: HyperParameters, memory: ReplayMemory, plotter: Plotter):
    results = list()

    for i_episode in range(hyper.episodes):
        # Reset the environment to a blank slate for each episode
        state = environment.reset()
        state = torch.tensor(state, dtype=torch.float32, device=hyper.device).unsqueeze(0)

        for i_step in range(hyper.steps):
            # Select an action to take at this step
            action = agent.select_action(state)

            # Impact of action on the enviromnet and corresponding reward
            next_state, rewards = environment.step(action.item())

            # Make pytorch friendly
            rewards = torch.tensor(rewards, device=hyper.device)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=hyper.device).unsqueeze(0)
            
            # Save action to the memory
            trans = Transition(state, action, next_state, rewards)
            memory.push(trans)

            # Update state
            state = next_state

            # Update policy network
            agent.optimize_policy(memory)

            # Update target network
            agent.optimize_target()

        # Save end state of episode to results
        results.append(state.item())

        # Plot results of episode
        plotter.plot(results,
                     agent.exploit_actions,
                     agent.explore_actions,
                     agent.policy_network,
                     hyper,
                     i_episode,
                     False)

    # Plot results of training
    plotter.plot(results,
                    agent.exploit_actions,
                    agent.explore_actions,
                    agent.policy_network,
                    hyper,
                    hyper.episodes,
                    True)

    return results

    