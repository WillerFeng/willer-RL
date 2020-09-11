import gym, os, gc
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

env_name = 'CartPole-v0'
env = gym.make(env_name)
env = env.unwrapped

env.seed(0)
utils.set_random_seed(0)

state_space  = env.observation_space.shape[0]
action_space = env.action_space.n

    
class REINFORCE(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        discount=0.97,
        tau=0.005,
        lr=3e-4,
        weight_decay=1e-4
        ):

        self.writer = SummaryWriter("runs/REINFORCE_"+ env_name + "_" +str(datetime.datetime.now()))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = net.Actor_RAM(state_dim, action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.discount = discount
        self.total_it = 0


    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state)
        m = Categorical(action)
        action = m.sample()
        return action.item(), m.log_prob(action)

    
    def train(self, episode):
        self.total_it += 1
        
        reward, log_action = zip(*episode)
        
        reward = utils.reward_shape(np.array(reward), self.discount)
        reward = torch.FloatTensor(reward).to(self.device)
        log_action = torch.stack(log_action)
        
        actor_loss = -(log_action * reward).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.writer.add_scalar('actor_loss', actor_loss, self.total_it)
        gc.collect()

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))

    
if __name__ == '__main__':
    
    episodes  = 600
    threshold = 1000
    
    agent = REINFORCE(state_space, action_space)
    print("<<=== Begin Train ===>>")
    for i_episode in range(episodes):

        total_reward = 0
        t = 0
        episode = []
        state = env.reset()
        for t in count():
            action, log_action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            episode.append([reward, log_action])
            state = next_state
            total_reward += reward
            if done or t >= threshold:
                break
              
        agent.train(episode)
        agent.writer.add_scalar('reward', total_reward, i_episode)

    print("<<=== Finish ===>>")