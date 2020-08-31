import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
import datetime
from tensorboardX import SummaryWriter
from itertools import count
from collections import namedtuple

from common import atari_wrappers, buffer, net, utils

env_name = 'CartPole-v0'
env = gym.make(env_name)
env = env.unwrapped

env.seed(0)
utils.set_random_seed(0)

state_space = env.observation_space.shape[0]
action_space = env.action_space.n


class Q_Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_Net, self).__init__()

        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, action_dim)

    def forward(self, state):

        q = F.relu(self.l1(state))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


class DQN():
    def __init__(
        self,
        state_dim,
        action_dim,
        epsilon=0.1,
        memory_capacity=2e4,
        discount=0.97,
        lr=3e-4,
        weight_decay=2e-3,
        batch_size=256,
        step_update_target_net=20,
    ):

        super(DQN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.eval_net   = Q_Net(state_dim, action_dim).to(self.device)
        self.target_net = Q_Net(state_dim, action_dim).to(self.device)

        self.memory    = buffer.ReplayBuffer(memory_capacity)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr, weight_decay=weight_decay)

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.discount = discount
        self.step_update_target_net = step_update_target_net
        self.learn_step_counter = 0
        self.epsilon = epsilon
        self.batch_size = batch_size


    def select_action(self, state):

        if np.random.randn() <= self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].cpu().numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.action_dim)
        return action


    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)


    def train(self):


        if self.learn_step_counter % self.step_update_target_net ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        self.epsilon *= 1.005

        batch_state, batch_action, batch_reward, batch_next_state, _ = self.memory.sample(self.batch_size)
        batch_state  = torch.FloatTensor(batch_state).to(self.device)
        batch_action = torch.LongTensor(batch_action).view(-1, 1).to(self.device)
        batch_reward = torch.FloatTensor(batch_reward).view(-1, 1).to(self.device)
        batch_next_state = torch.FloatTensor(batch_next_state).to(self.device)


        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + self.discount * q_next.max(1)[0].view(self.batch_size, 1)

        loss = F.mse_loss(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def main():

    writer = SummaryWriter("runs/Baseline-DQN_"+ env_name + "_" +str(datetime.datetime.now()))
    episodes  = 800
    threshold = 1000
    agent = DQN(state_space, action_space)

    print("<<=== Begin Train ===>>")
    for i_episode in range(episodes):

        total_reward = 0
        t = 0
        state = env.reset()
        for t in count():

            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done or t >= threshold:
                break

        agent.train()
        writer.add_scalar('reward', total_reward, i_episode)
    print("<<=== Finish ===>>")


if __name__ == '__main__':
    main()
