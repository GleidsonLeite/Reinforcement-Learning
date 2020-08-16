from math import exp

class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        self.current_step = 0

    def get_exploration_rate(self, updateExplorationRate:bool):
        exploration_rate = self.end + (self.start - self.end) * exp(-1 * self.current_step * self.decay)
        if updateExplorationRate:
            self.update_step()
        return exploration_rate
    
    def update_step(self):
        self.current_step += 1
    
    def clear_steps(self):
        self.current_step = 0