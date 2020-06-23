# Terminal Estmation
import gym, os
import numpy as np
import datetime
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

from common import atari_wrappers, buffer, net, utils

env = gym.make('CartPole-v0')
env = env.unwrapped

env.seed(0)
utils.set_random_seed(0)

state_space = env.observation_space.shape[0]
action_space = env.action_space.n

learning_rate = 0.02
gamma = 0.95
weight_decay = 1e-3
render = False
eps = np.finfo(np.float32).eps.item()
SavedAction = namedtuple('SavedAction', ['log_prob', 'q_value', 't_value'])


class Policy(nn.Module):
    def __init__(self, state_space:int, action_space:int, net_type:str, hidden_size:int=256):       
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        self.fc_q_value = nn.Linear(hidden_size, 1)
        self.fc_t_value = nn.Linear(hidden_size, 1)
        self.fc_action = nn.Linear(hidden_size, action_space)
        
        if   net_type == 'value':
            self.last_active = nn.ReLU()
        elif net_type == 'policy':
            self.last_active = nn.Softmax(dim=-1)
        else:
            raise ValueError("Undefined net type")
            
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        action  = self.last_active(self.fc_action(x))
        q_value = self.fc_q_value(x)
        t_value = self.fc_t_value(x)
        
        return action, q_value, t_value
                  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Policy(state_space, action_space, 'policy', 128).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
save_actions, env_rewards = [], []

def select_action(state):
    
    global save_actions
    state = torch.from_numpy(state).double().to(device)
    probs, q_value, t_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    
    save_actions.append(SavedAction(m.log_prob(action), q_value, t_value))

    return action.item()


def finish_episode():
    global save_actions, env_rewards
    
    R = 0
    save_actions = save_actions
    policy_loss = []
    q_value_loss = []
    t_value_loss = []
    rewards = []

    for r in env_rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)

    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
                               
    for (log_prob , q_value, t_value), r in zip(save_actions[:-1], rewards[:-1]):
        reward = r - (q_value.item() + t_value.item())
        policy_loss.append(-log_prob * reward)
        q_value_loss.append(F.smooth_l1_loss(q_value, torch.tensor([r]).to(device)))
    t_value_loss.append(F.mse_loss(save_actions[-1].t_value, torch.tensor([rewards[-1]]]).to(device)))
    
    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum() + torch.stack(q_value_loss).sum() + torch.stack(t_value_loss).sum()
    loss.backward()
    optimizer.step()

    save_actions, env_rewards = [], []

    
def main():

    live_time = []
    writer = SummaryWriter("runs/TE-AC_decay=1e-3_CartPole-v0_"+str(datetime.datetime.now()))
    episodes  = 400
    threshold = 2000
    
    print("Collecting Experience....")
    for i_episode in range(episodes):
        state = env.reset()
        for t in count():
            action = select_action(state)
            state, reward, done, info = env.step(action)
            if render: 
                env.render()
            env_rewards.append(reward)

            if done or t >= threshold:
                break
    
        writer.add_scalar('live_time', t, i_episode)
        finish_episode()
    print("finish")

if __name__ == '__main__':
    main()


