class Plotter(object):
    def __init__(self, plot_function, frequency=30):
        self.plot_function = plot_function
        self.frequency = frequency

    def plot(self, states, exploits, explores, policy_net, hyper, i_episode, show_result=False):
        if i_episode % self.frequency != 0:
            return
        
        self.plot_function(states, 
                            exploits, 
                            explores, 
                            policy_net, 
                            hyper, 
                            i_episode, 
                            show_result)