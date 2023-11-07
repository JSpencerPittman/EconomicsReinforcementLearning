from singleagent.train import train_agent
from singleagent.agent import SingleAgent
from singleagent.utils import ( Environment,
                                HyperParameters,
                                ReplayMemory,
                                Plotter )
from enum import Enum
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
import torch

class Actions(Enum):
    WORK = 0
    SLEEP = 1

class WorkOrSleep(Environment):
    def __init__(self, actions: Enum):
        super(WorkOrSleep, self).__init__(actions)
        self.state = [0]

    def reset(self):
        self.state = [0]
        return self.state
    
    def step(self, action):
        next_state = self.state
        rewards = []

        if action == Actions.WORK.value:
            next_state[0] += 1
            rewards.append(1)
        else:
            rewards.append(-1)
        
        self.state = next_state

        return next_state, rewards

def plot_progress(states, exploits, explores, policy_net, hyper, i_episode, show_result):
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

    ax_tm.bar([0,1], exploits, tick_label=["work", "sleep"])
    ax_tm.set_xticks([0,1])
    ax_tm.set_xlabel("Action")

    ax_tr.bar([0,1], explores, tick_label=["work", "sleep"])
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

params = HyperParameters(episodes=100, 
                         steps=10,
                         batch_size=16, 
                         eps_decay=4000, 
                         gamma=0.9, 
                         learning_rate=1e-3)
environment = WorkOrSleep(Actions)
plotter = Plotter(plot_progress)
agent = SingleAgent(environment, params)
memory = ReplayMemory(1000)

print(train_agent(agent, environment, params, memory, plotter))

