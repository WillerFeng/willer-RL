import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
from tensorboardX import SummaryWriter
from collections import namedtuple
np.set_printoptions(precision=4)

class Maze:
    def __init__(self, size, random_init=False):
        self.size = size
        self.pos  = None
        self.random_init = random_init
    def reset(self):
        if self.random_init:
            self.pos = random.randint(0, self.size-1)
        else:
            self.pos = 0
        return self.pos

    def step(self, action):
        if action == 0 or action == 1:
            return self.pos, -10, True
        if action == 2:
            if self.pos == 0:
                return self.pos, -10, True
            self.pos -= 1
            return self.pos, -0.1, False
        if action == 3:
            if self.pos == self.size - 1:
                return self.size, 10, True
            self.pos += 1
            return self.pos, -0.1, False

    def render(self):
        env = ['-'] * (self.size + 1)
        env[-1] = '>'
        env[self.pos] = '@'
        str_env = ''.join(env)
        print(str_env)

    def get_best_q(self):
        best_q = np.zeros((self.size, 4))
        for i in range(self.size):
            best_q[i] = np.array([-10, -10, 10 - (self.size - i + 1) * 0.1, 10 - (self.size - i - 1) * 0.1])
        best_q[0][2] = -1
        return best_q

class Q_table:
    def __init__(self, action_space, state_space):
        #self._q = np.zeros((state_space, action_space))
        self._q = np.random.rand(state_space, action_space)
        self.action_space = action_space
        self.epsilon = 0.0
        self.gamma = 1
        self.alpha = 0.1
        self.learn_step_counter = 0

    def action(self, state):

        return random.choices(range(self.action_space), weights=np.exp(self._q[state]))[0]

        if random.random() <= self.epsilon:
            return self._q[state].argmax()
        else:
            return random.randint(0, self.action_space-1)

    def update(self, state, action, reward, nex_state, done):
        if not done:
            delta = reward + max(self._q[nex_state]) - self._q[state][action]
        else:
            delta = reward - self._q[state][action]
        self._q[state][action] += self.alpha * delta
        self.learn_step_counter += 1

        if self.learn_step_counter % 800 == 0:
            self.epsilon += 0.1
        self.epsilon = min(1, self.epsilon)

class DHDB_table:
    def __init__(self, action_space, state_space):
#         self._q = np.zeros((state_space, action_space))
#         self._t = np.zeros((state_space, action_space))
        self._q = np.random.rand(state_space, action_space)
        self._t = np.random.rand(state_space, action_space)
        self.action_space = action_space
        self.epsilon = 0.0
        self.gamma = 1
        self.alpha = 0.1
        self.beta = 0.8
        self.learn_step_counter = 0

    def action(self, state):
        return random.choices(range(self.action_space), weights=np.exp(self._q[state] + self._t[state]) )[0]

        if random.random() <= self.epsilon:
            return (self._q[state]+self._t[state]).argmax()
        else:
            return random.randint(0, self.action_space-1)

    def update(self, state, action, reward, nex_state, done):
        if not done:
            delta = reward + max(self._q[nex_state] + self._t[nex_state]) - self._q[state][action]
            self._q[state][action] += self.alpha * delta
            self._t[state][action]  = self.beta * self._t[state][action]
        else:
            self._t[state][action] += self.alpha * (reward - self._t[state][action])
            self._q[state][action]  = self.beta * self._q[state][action]
            #self._t[state][action] = self.beta * self._t[state][action] + (1-self.beta) * reward
        self.learn_step_counter += 1

        if self.learn_step_counter % 800 == 0:
            self.epsilon += 0.1
        self.epsilon = min(1, self.epsilon)



if __name__ == '__main__':
    state_size = 30
    env = Maze(state_size)
    episode = 12000
    render = False


    agent = Q_table(4, state_size)
    writer = SummaryWriter("runs/Qt-" + str(datetime.datetime.now()))
    best_q = env.get_best_q()
    for i in range(episode):
        state = env.reset()
        ep_reward = 0
        while True:
            if render:
                env.render()
            action = agent.action(state)
            nex_state, reward, done = env.step(action)
            agent.update(state, action, reward, nex_state, done)
            state = nex_state
            ep_reward += reward
            if done:
                break
        writer.add_scalar("reward", ep_reward, i)
        writer.add_scalar("loss", np.sum(np.abs(best_q - agent._q)), i)
    Qt_q = agent._q


    agent = DHDB_table(4, state_size)
    writer = SummaryWriter("runs/TE-" + str(datetime.datetime.now()))
    for i in range(episode):
        state = env.reset()
        ep_reward = 0
        while True:
            if render:
                env.render()
            action = agent.action(state)
            nex_state, reward, done = env.step(action)
            agent.update(state, action, reward, nex_state, done)
            state = nex_state
            ep_reward += reward
            if done:
                break
        writer.add_scalar("reward", ep_reward, i)
        writer.add_scalar("loss", np.sum(np.abs(best_q - agent._q - agent._t)), i)
    DHDB_q = agent._q
    DHDB_t = agent._t
    print("Finish")
