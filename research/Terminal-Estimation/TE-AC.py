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

env_name = 'MountainCar-v0'
env = gym.make(env_name)
env = env.unwrapped

env.seed(0)
utils.set_random_seed(0)

state_space = env.observation_space.shape[0]
action_space = env.action_space.n

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, action_dim)


    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return F.softmax(self.l3(a), dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + 1, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TE_AC(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        discount=0.97,
        lr=3e-4,
        weight_decay=2e-3,
        batch_size=128
        ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.q_critic = Critic(state_dim, action_dim).to(self.device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=lr, weight_decay=weight_decay)

        self.t_critic = Critic(state_dim, action_dim).to(self.device)
        self.t_critic_optimizer = torch.optim.Adam(self.t_critic.parameters(), lr=lr, weight_decay=weight_decay)
        self.t_buffer = buffer.ReplayBuffer(size=1e4)

        self.discount = discount
        self.batch_size = batch_size
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state)
        m = Categorical(action)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def train(self, episode):

        self.total_it += 1

        state, action, reward, next_state, log_action = zip(*episode)
        reward = utils.reward_shape(np.array(reward), self.discount)
        end_state, end_action, end_reward, _, _ = self.t_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        log_action = torch.stack(log_action)

        end_state  = torch.FloatTensor(end_state).to(self.device)
        end_action = torch.FloatTensor(end_action).to(self.device)
        end_reward = torch.FloatTensor(end_reward).to(self.device)

        t_critic_loss = F.mse_loss(end_reward, self.t_critic(end_state, end_action))
        self.t_critic_optimizer.zero_grad()
        t_critic_loss.backward()
        self.t_critic_optimizer.step()


        with torch.no_grad():
            next_action = self.actor(next_state)
            m = Categorical(next_action)
            action_sample = m.sample().cpu().numpy().reshape(-1, 1)
            action_sample = torch.FloatTensor(action_sample).to(self.device)

            target_q = self.q_critic(next_state, action_sample)
            target_t = self.t_critic(next_state, action_sample)
            target_value = reward + self.discount * (target_q + target_t)

        q_critic_loss = F.mse_loss(target_value, self.q_critic(state, action))
        self.q_critic_optimizer.zero_grad()
        q_critic_loss.backward()
        self.q_critic_optimizer.step()

        actor_loss = -(log_action * (reward - self.q_critic(state, action) - self.t_critic(state, action))).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        gc.collect()

    def save(self, filename):
        torch.save(self.q_critic.state_dict(), filename + "_Q_critic_" + str(datetime.datetime.now()))
        torch.save(self.q_critic_optimizer.state_dict(), filename + "_Q_critic_optimizer_" + str(datetime.datetime.now()))

        torch.save(self.t_critic.state_dict(), filename + "_T_critic_" + str(datetime.datetime.now()))
        torch.save(self.t_critic_optimizer.state_dict(), filename + "_T_critic_optimizer_" + str(datetime.datetime.now()))

        torch.save(self.actor.state_dict(), filename + "_actor_" + str(datetime.datetime.now()))
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer_" + str(datetime.datetime.now()))


    def load(self, filename):
        self.q_critic.load_state_dict(torch.load(filename + "_Q_critic"))
        self.q_critic_optimizer.load_state_dict(torch.load(filename + "_Q_critic_optimizer"))

        self.t_critic.load_state_dict(torch.load(filename + "_T_critic"))
        self.t_critic_optimizer.load_state_dict(torch.load(filename + "_T_critic_optimizer"))

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))


def main():

    writer = SummaryWriter("runs/TE-AC_2e-3"+ env_name + "_" +str(datetime.datetime.now()))
    episodes  = 400
    threshold = 1000
    agent = TE_AC(state_space, action_space)

    print("<<=== Begin Train ===>>")
    for i_episode in range(episodes):

        total_reward = 0
        t = 0
        episode = []
        state = env.reset()
        for t in count():
            action, log_action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            episode.append([state, [action], [reward], next_state, log_action])
            state = next_state
            total_reward += reward
            if done or t >= threshold:
                agent.t_buffer.add(state, [action], [reward], None, None)
                break

        agent.train(episode)
        writer.add_scalar('reward', total_reward, i_episode)

    print("<<=== Finish ===>>")

if __name__ == '__main__':
    main()
