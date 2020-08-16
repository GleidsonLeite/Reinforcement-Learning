from collections import deque
import os

from tqdm import tqdm
import gym
from gym.wrappers import Monitor
import numpy as np
from matplotlib import pyplot as plt

from torch import optim

from replay_memory import ReplayMemory
from strategy import EpsilonGreedyStrategy
from model import Model
from agent import Agent
from model_utils import saveModel, loadModel

# Environment Settings
env = gym.make('LunarLander-v2')
env.seed(0)

state_space = env.observation_space.shape[0]
action_space = env.action_space.n
print('State shape: ', state_space)
print('Number of actions: ', action_space)

# ReplayMemory Settings
capacity = 5000
batch_size = 64
replayMemory = ReplayMemory(capacity, batch_size)

# Strategy Settings
strategy = EpsilonGreedyStrategy(1, 0, 1E-3)    

# Deep Learning Model
model = Model(state_space, action_space)
path = './checkpoint.pth.tar' # To save/load the model

pathExist = os.path.isfile(path)
pathExist and input('The file {} will be replaced, do you wish to continue? (if not, press ctrl+c)'.format(path))
# if you want load some model uncoment the lines below
not(pathExist) and print('{} don\'t founded'.format(path))
model, _, _, _ = loadModel(path, model) if pathExist else (model, *(None for i in range(3)))

learning_rate = 1E-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Agent Settings
updateEvery = 4
agent = Agent(strategy, replayMemory, model, optimizer, updateEvery, .98, 1E-3)
strategy.current_step=5000
'''

# Training
n_episodes = 5000
n_steps = 1000

# Score

window_size = 100
scores_window = deque(maxlen=window_size)
scores = []

episodes = tqdm(range(n_episodes))

try:
    for episode in episodes:
        state = env.reset()
        score = 0
        for step in range(n_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score+=reward
            if done:
                break
        strategy.update_step()


        scores_window.append(score)
        mean_score_window = np.mean(scores_window)
        scores.append(mean_score_window)
        episodes.set_description('Average Score = {:.2f}\tExploration rate = {:.2f}'.format(mean_score_window, strategy.get_exploration_rate(False)))
except:
    pass

print('\nSaving the model checkpoint as {}'.format(path))
saveModel(model, path)
scores = np.asarray(scores)
np.save('scores.npy', scores)
plt.plot(scores)
plt.show()
'''
env = Monitor(env, './video', video_callable=lambda episode_id: True, force=True)
state = env.reset()
while True:
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    agent.step(state, action, reward, next_state, done)
    state = next_state
    if done: break

env.close()