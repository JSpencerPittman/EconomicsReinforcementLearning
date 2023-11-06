import torch
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np

from agent import Agent
from environment import Environment
from memory import ReplayMemory, Transition

from time import sleep

class HyperParameters(object):
    def __init__(self, 
                 episodes=100, 
                 gamma=0.99,
                 tau=0.005,
                 eps_start=0.9,
                 eps_end=0.05,
                 eps_decay=1000,
                 learning_rate=1e-4,
                 batch_size=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.episodes = episodes
        self.gamma = gamma
        self.tau = tau
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size


def plot_preference(network, hyper, i_episode=-1, show_result=False):
    states = np.array([i*10 for i in range(10)])
    states_gpu = torch.tensor(states, dtype=torch.float32, device=hyper.device).unsqueeze(1)
    
    with torch.no_grad():
        preferences = network(states_gpu).cpu().numpy()

    fig = plt.figure(2, clear=True, figsize=(8,4))

    if show_result:
        plt.title('Preference: Completed')
    else:
        plt.title(f'Preference: Training({i_episode})')

    plt.bar(states, preferences[:,0], width=2, label="Work")
    plt.bar(states+2, preferences[:,1], width=2, label="Sleep")
    plt.xticks(states)
    plt.legend()

    plt.pause(0.1)

def plot_progress(states, exploit, explore, policy_net, hyper, i_episode=-1, show_result=False):
    fig = plt.figure(1, clear=True, figsize=(12,10))

    grid_size = (2,3)
    ax_tl = plt.subplot2grid(grid_size, (0, 0), fig=fig)
    ax_tm = plt.subplot2grid(grid_size, (0, 1), fig=fig)
    ax_tr = plt.subplot2grid(grid_size, (0, 2), fig=fig)
    ax_bt = plt.subplot2grid(grid_size, (1, 0), colspan=3, fig=fig)

    if show_result:
        ax_tl.set_title('State: Completed')
        ax_tm.set_title('Exploit: Completed')
        ax_tr.set_title('Explore: Completed')
    else:
        ax_tl.set_title(f'State: Training({i_episode})')
        ax_tm.set_title(f'Exploit: Training({i_episode})')
        ax_tr.set_title(f'Explore: Training({i_episode})')
    
    ax_tl.set_ylabel("Money")
    try:
        smoothed = savgol_filter(states, window_length=30, polyorder=2)
        ax_tl.plot(smoothed)
    except ValueError:
        ax_tl.plot(states)

    ax_tm.bar([0,1], exploit, tick_label=["work", "sleep"])
    ax_tm.set_xticks([0,1])
    ax_tm.set_xlabel("Action")

    ax_tr.bar([0,1], explore, tick_label=["work", "sleep"])
    ax_tr.set_xticks([0,1])
    ax_tr.set_xlabel("Action")

    states = np.array([i*10 for i in range(10)])
    states_gpu = torch.tensor(states, dtype=torch.float32, device=hyper.device).unsqueeze(1)
    with torch.no_grad():
        preferences = policy_net(states_gpu).cpu().numpy()

    if show_result:
        ax_bt.set_title('Preference: Completed')
    else:
        ax_bt.set_title(f'Preference: Training({i_episode})')

    ax_bt.bar(states, preferences[:,0], width=2, label="Work")
    ax_bt.bar(states+2, preferences[:,1], width=2, label="Sleep")
    ax_bt.set_xticks(states)
    ax_bt.legend()

    plt.pause(1)

def train_agent(agent: Agent, environment: Environment, hyper: HyperParameters, memory: ReplayMemory):
    results = list()

    willing_work, willing_sleep = 0, 0

    for i_episode in range(hyper.episodes):  # Number of episodes
        state = environment.reset()
        state = torch.tensor(state, dtype=torch.float32, device=hyper.device).unsqueeze(0)

        for t in range(100):
            action = agent.select_action(state)

            if action.item() == 0:
                willing_work += 1
            else:
                willing_sleep += 1

            next_state, rewards = environment.step(action)
            rewards = torch.tensor(rewards, device=hyper.device)
            
            next_state = torch.tensor(next_state, dtype=torch.float32, device=hyper.device).unsqueeze(0)
            
            trans = Transition(state, action, next_state, rewards)
            memory.push(trans)

            state = next_state

            agent.optimize_model(memory)
            agent.update_target_network()

        results.append(state.item())

        if i_episode % 30 == 0:
            plot_progress(results, agent.exploit, agent.explore, agent.policy_network, hyper, i_episode=i_episode)

    plot_progress(results, agent.exploit, agent.explore, agent.policy_network, hyper, show_result=True)
    plt.ioff()
    plt.show()

    return results

    