from model import Model
from strategy import EpsilonGreedyStrategy
from replay_memory import ReplayMemory

from model_utils import takeActionWithModel, convertBatchToTensor, updatePolicy

import random

class Agent:
    """
        This class will be the agent used in DQN Algorithm  
    """
    def __init__(self, Strategy:EpsilonGreedyStrategy, 
                        Memory:ReplayMemory, model:Model,
                        optimizer, updateEvery:int, gamma, tau):

        self.strategy = Strategy
        self.memory = Memory

        self.policy_net = model
        self.target_net = Model(self.policy_net.n_input, self.policy_net.n_output)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optimizer

        self.gamma = gamma
        self.tau = tau

        self.updateEvery = updateEvery
        self.currentStep = 0


    def act(self, state, updateExploration=False):
        exploration_rate = self.strategy.get_exploration_rate(updateExploration)
        if random.random()<exploration_rate:
            return random.randrange(self.policy_net.n_output)
        else:
            return takeActionWithModel(state, self.policy_net)
    
    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.currentStep+=1
        if self.currentStep%self.updateEvery==0:
            self.currentStep=0
            samples = self.memory.sample()
            if samples:
                updatePolicy(self.policy_net, self.target_net, self.optimizer, samples, self.gamma, self.tau)


