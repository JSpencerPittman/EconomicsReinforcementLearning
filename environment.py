class Environment:
    def __init__(self):
        self.state = [0]

    def reset(self):
        self.state = [0]
        return self.state

    def step(self, action):
        next_state = self.state.copy()
        rewards = []
        
        if action == 0:
            next_state[0] += 1
            rewards.append(1)
        else:
            rewards.append(0)

        self.state = next_state

        return next_state, rewards
