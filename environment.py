class Environment:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.state = [0] * num_agents
        
    def reset(self):
        self.state = [0] * self.num_agents
        return self.state

    def step(self, actions):
        # Apply actions to the environment and return new state, reward, and done status
        # In this case, action 0 might be 'work' and action 1 'sleep'
        next_state = self.state.copy()
        rewards = []
        done = False
        
        for i, action in enumerate(actions):
            if action == 0:  # 'work'
                next_state[i] += 1  # Increase MONEY
                rewards.append(1)
            elif action == 1:  # 'sleep'
                rewards.append(0)  # No MONEY increase

        return next_state, rewards, done
