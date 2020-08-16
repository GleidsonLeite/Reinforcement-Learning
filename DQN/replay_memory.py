from collections import namedtuple, deque
import random

class ReplayMemory:

    def __init__(self, capacity:int, batch_size:int):
        self.experience = namedtuple(
            'Experience', ('State', 'Action', 'Reward', 'Next_state', 'Done')
            )
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
    
    def push(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def sample(self, nSamples=None):
        batch_size = nSamples if nSamples else self.batch_size
        if len(self.memory)>=batch_size:
            return random.sample(self.memory, batch_size)
        else:
            #print('The Replay Memory don\'t have enough data! Please push more data before get a sample')
            return None

