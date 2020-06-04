import math
import random

import gym
import numpy as np
import datetime
import gc
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as _mp

from torch.distributions import Categorical
from tensorboardX import SummaryWriter
from collections import namedtuple


SavedAction = namedtuple('SavedAction', ['log_prob', 'q_value'])

class Policy(nn.Module):
    
    def __init__(self, state_space, action_space, hidden_size=64):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_space, hidden_size)

        self.action_head = nn.Linear(hidden_size, action_space)
        self.value_head  = nn.Linear(hidden_size, 1) 

        self.save_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_score = self.action_head(x)
        state_value  = self.value_head(x)

        return F.softmax(action_score, dim=-1), state_value

class GlobalAdam(torch.optim.Adam):
    def __init__(self, params, lr):
        super(GlobalAdam, self).__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
        
# need cuda support
class Actor:
    def __init__(self, env_name, global_actor=None, global_optim=None, max_epochs=600, center=False):
        
        
        self.env = gym.make(env_name)
        self.net = Policy(self.env.observation_space.shape[0], self.env.action_space.n, hidden_size=64)
        self.threshold = 2000
        
        if not center:
            self.center       = global_actor
            self.global_net   = global_actor.net
            self.global_optim = global_optim
            self.save_actions = []
            self.rewards = []
            self.episodes = max_epochs
            self.net.load_state_dict(self.global_net.state_dict())
            

        else:
            self.count = 0
            self.global_count = 0
            self.writer = SummaryWriter("runs/A3C_" + str(datetime.datetime.now()))
            self.net.share_memory()
            

    def action(self, state, train=True):
        
        state = torch.from_numpy(state).float()
        probs, state_value = self.net(state)
        m      = Categorical(probs)
        action = m.sample()
        if train:
            self.save_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()
        
        
    def rollout(self):
        for i_episode in range(self.episodes):
            state = self.env.reset()
            for t in count():
                action = self.action(state)
                state, reward, done, info = self.env.step(action)
                self.rewards.append(reward)

                if done or t >= self.threshold:
                    break
            self.train()
            
            
    def train(self):
        
        R = 0
        gamma = 0.95
        eps = np.finfo(np.float32).eps.item()
        
        save_actions = self.save_actions
        policy_loss = []
        value_loss = []
        rewards = []

        for r in self.rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)

        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

        for (log_prob , value), r in zip(save_actions, rewards):
            reward = r - value.item()
            policy_loss.append(-log_prob * reward)
            value_loss.append(F.smooth_l1_loss(value, torch.tensor([r])))
            
        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
        loss.backward()
        

        self.global_optim.zero_grad()
        for local_param, global_param in zip(self.net.parameters(), self.global_net.parameters()):
            global_param._grad = local_param.grad

        self.global_optim.step()
        self.net.load_state_dict(self.global_net.state_dict())

        
        self.rewards      = []
        self.save_actions = []
        gc.collect()
    
    def test_performance(self):
        
        state = self.env.reset()
        t = 0
        for t in count():
            action = self.action(state, False)
            state, reward, done, info = self.env.step(action)
            if done or t >= self.threshold:
                break
        self.writer.add_scalar("live_time", t, self.count)
        print(self.count, t)
        self.count += 1
            


# Single Process Worker     
def single(env_name, global_actor, global_optim, max_epochs=600):
    
    actor = Actor(env_name, global_actor, global_optim, max_epochs=600)
    actor.rollout()   
    
def test(global_actor):
    global_actor.test_performance()
              
def main(num_processes=2):
    
    env_name = "CartPole-v0"
    learning_rate = 1e-3
    
    global_actor = Actor(env_name, center=True)
    global_optim = GlobalAdam(global_actor.net.parameters(), lr=learning_rate)
    
    #mp = _mp.get_context("spawn")
    processes = []
    for i in range(num_processes):
        process = _mp.Process(target=single, args=(env_name, global_actor, global_optim, 100))
        process.start()
        processes.append(process)
        
    process = _mp.Process(target=test, args=(global_actor, ))
    process.start()
    processes.append(process)
    for process in processes:
        process.join()
    
    print("finish")
    
if __name__ == '__main__':
    main()
