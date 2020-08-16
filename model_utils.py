import torch
import torch.nn.functional as F

import numpy as np


def convertBatchToTensor(batch):
    """
        This function will convert a bath of experiences in Torch tensor

        Parameters
        ----------
        batch: a list of namedtuple

        Return
        ------
        torch tensor
    """
    ('State', 'Action', 'Reward', 'Next_state', 'Done')
    states = torch.from_numpy(np.vstack([e.State for e in batch if e is not None])).float()
    actions = torch.from_numpy(np.vstack([e.Action for e in batch if e is not None])).long()
    rewards = torch.from_numpy(np.vstack([e.Reward for e in batch if e is not None])).float()
    next_states = torch.from_numpy(np.vstack([e.Next_state for e in batch if e is not None])).float()
    dones = torch.from_numpy(np.vstack([e.Done for e in batch if e is not None]).astype(np.uint8)).float()
    return (states, actions, rewards, next_states, dones)


def takeActionWithModel(state, policy):
    """
        This method will get an action from a policy

        Parameters
        ----------
        state: a list with each parameter of a state
        policy: a torch model who will get a state and will predict the actions from that state

        Return
        ------
        action: a integer value specifying an action to take.
    """
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        actions = policy(state)
    return torch.argmax(actions).item()

def updatePolicy(policy, target, optimizer, experiences, gamma, tau):
    """
        This method will update the Policy's parameter.

        The way used to update the policy model is described on DQN paper.

        The target is:
            reward_j if the episode end on the next step
            reward_j + gamma*argmax(target(next_states))
        
        The error is calculated by Mean Squared Error:
            error = MSE(target - policy(states))

        Parameters
        ----------
        policy: A torch model

        target: A torch model

        optimizer: An optimizer which are with the policy parameters associated

        experiences: A batch of experiences
        
        gamma: The discount factor to use in DQN algorithm
    """
    states, actions, rewards, next_states, dones = convertBatchToTensor(experiences)
    Q_targets_next = target(next_states).detach().max(1)[0].unsqueeze(1)
    Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
    Q_expected = policy(states).gather(1, actions)
    loss = F.mse_loss(Q_expected, Q_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    softUpdateTarget(policy, target, tau)

def softUpdateTarget(policy, target, tau):
    """
        This function will update the Target model with soft update method

        Parameters
        ----------
        
        policy: A torch model
        
        target: A torch model
        
        tau: update factor
    """
    for target_param, local_param in zip(target.parameters(), policy.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

def saveModel(model, PATH, optimizer=None, epoch=None, loss=None):
    """
        This function will save a checkpoint of your model and another parameters like epoch and loss.

        Parameters
        ----------
        
        model: torch model

        PATH: The PATH where you would like to save the checkpoint. Please, don't forget to set the file names (Example: checkpoint.pth.tar).
        
        epoch: An integer value of the current epoch you want to save.

        loss: The loss value yout want to save.

        PS: Don't have problem if you don't pass the epoch and loss arguments, they are configured to None by default.
    """
    torch.save({
        'epoch': epoch if epoch else None,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'loss': loss if loss else None
    }, PATH)

def loadModel(path, model, optimizer=None, epoch=None, loss=None, isEval=True):

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if epoch:
        epoch = checkpoint['epoch']
    if loss:
        loss = checkpoint['loss']

    model.eval() if isEval else model.train()

    return (model, optimizer, epoch, loss)