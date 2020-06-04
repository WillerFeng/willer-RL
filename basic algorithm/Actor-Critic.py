import gym, os
import numpy as np

from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

#Parameters
env = gym.make('CartPole-v0')
env = env.unwrapped

env.seed(1)
torch.manual_seed(1)

state_space = env.observation_space.shape[0]
action_space = env.action_space.n


#Hyperparameters
learning_rate = 0.01
gamma = 0.99
render = False
eps = np.finfo(np.float32).eps.item()
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_space, 32)

        self.action_head = nn.Linear(32, action_space)
        self.value_head = nn.Linear(32, 1) # Scalar Value

        self.save_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_score = self.action_head(x)
        state_value = self.value_head(x)

        return F.softmax(action_score, dim=-1), state_value

model = Policy()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.save_actions.append(SavedAction(m.log_prob(action), state_value))

    return action.item()


def finish_episode():
    R = 0
    save_actions = model.save_actions
    policy_loss = []
    value_loss = []
    rewards = []

    for r in model.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)

    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    for (log_prob , value), r in zip(save_actions, rewards):
        reward = r - value.item()
        policy_loss.append(-log_prob * reward)
        value_loss.append(F.smooth_l1_loss(value, torch.tensor([r])))

    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.save_actions[:]

def main():
    running_reward = 10
    live_time = []
    writer = SummaryWriter()
    episodes = 300
    for i_episode in range(episodes):
        state = env.reset()
        for t in count():
            action = select_action(state)
            state, reward, done, info = env.step(action)
            if render: 
                env.render()
            model.rewards.append(reward)

            if done or t >= 1000:
                break

        writer.add_scalar('live time', t, i_episode)
        finish_episode()
    print("finish")

if __name__ == '__main__':
    main()


