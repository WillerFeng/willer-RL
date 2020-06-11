import gym, os
import numpy as np
import gc
from itertools import count
from collections import namedtuple
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

from common import atari_wrappers, buffer, net, util


env = gym.make('CartPole-v0')
env = env.unwrapped

env.seed(0)
util.set_random_seed(0)

state_space = env.observation_space.shape[0]
action_space = env.action_space.n

learning_rate = 0.01
gamma = 0.95
weight_decay = 1e-4
render = False
eps = np.finfo(np.float32).eps.item()
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = net.Dueling_FullyNet(state_space, action_space, 'policy', hidden_size=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

save_actions, env_rewards = [], []

def select_action(state):
    global save_actions
    
    state = torch.from_numpy(state).float().to(device)
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    
    save_actions.append(SavedAction(m.log_prob(action), state_value))

    return action.item()


def finish_episode():
    global save_actions, env_rewards
    
    R = 0
    policy_loss = []
    value_loss = []
    rewards = []

    for r in env_rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
        
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    for (log_prob , value), r in zip(save_actions, rewards):
        reward = r - value.item()
        policy_loss.append(-log_prob * reward)
        value_loss.append(F.smooth_l1_loss(value, torch.tensor([r]).to(device)))

    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
    loss.backward()
    optimizer.step()

    save_actions = []
    env_reward = []
    gc.collect()

def main():

    live_time = []
    writer = SummaryWriter("runs/Baseline-AC_CartPole-v0_"+str(datetime.datetime.now()))
    episodes  = 400
    threshold = 2000
    global env_rewards
    
    print("Collecting Experience....")
    for i_episode in range(episodes):
        state = env.reset()
        total_reward = 0
        t = 0
        for t in count():
            action = select_action(state)
            state, reward, done, info = env.step(action)
            if render: 
                env.render()
            env_rewards.append(reward)
            if done:
                break

        writer.add_scalar('live_time', t, i_episode)
        finish_episode()
    print("finish")

if __name__ == '__main__':
    main()


