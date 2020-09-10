import gym, os, gc, time
import numpy as np
import datetime
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

from common import atari_wrappers, buffer, net, utils


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, epsilon=0.9):
        super(DQN, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon

        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, action_dim)

    def forward(self, state):

        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.l3(a)

    def select_action(self, state):
        if np.random.randn() <= self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            action_value = self.forward(state)
            action = torch.max(action_value, 1)[1].cpu().numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.action_dim)
        return action

class APE_X:
    def __init__(
        self,
        env_name,
        state_dim,
        action_dim,
        num_actor=4,
        num_learner=1,
        batch_size=128,
        discount=0.97,
        tau=0.005,
        lr=3e-4,
        memory_capacity=1e5,
    ):
        self.num_actor = num_actor
        self.num_learner = num_learner
        self.queue = mp.Queue()

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cpu')

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env_name = env_name
        self.lr = lr
        self.discount = discount
        self.batch_size = batch_size

        self.center_agent = DQN(self.state_dim, self.action_dim)
        self.center_agent.share_memory()
        self.buffer = buffer.ReplayBuffer(memory_capacity)


    def train(self):

        print("<<=== Begin Train ===>>")

        process_pool = []
        process_prime = mp.Process(target=self.run_actor, args=(self.queue, self.center_agent), kwargs={'center':True, 'record_performance':True})
        process_prime.start()
        process_pool.append(process_prime)

        for i in range(self.num_actor-1):
            process_sub = mp.Process(target=self.run_actor, args=(self.queue, ))
            process_sub.start()
            process_pool.append(process_sub)

        process_learn = mp.Process(target=self.run_learner, args=(self.center_agent, self.queue, ))
        process_learn.start()
        process_pool.append(process_learn)

        for process in process_pool:
            process.join()

        print("<<=== Finish ===>>")

    def run_actor(self, experience_queue, agent=None, train_epoch=400, center=False, record_performance=False):

        if not agent:
            agent = DQN(self.state_dim, self.action_dim)
        if record_performance:
            writer = SummaryWriter("runs/APE-X_n-core_"+ str(self.num_actor) + '_' + self.env_name + "_" +str(datetime.datetime.now()))

        env = gym.make(self.env_name)
        env = env.unwrapped
        env.seed(0)
        utils.set_random_seed(0)

        threshold = 1000

        for epoch in range(train_epoch):
            total_reward = 0
            episode = []
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)

                self.buffer.add(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done or t >= threshold:
                    break

            if record_performance:
                experience_queue.put(epoch)
                writer.add_scalar('reward', total_reward, epoch)

            if not center and epoch % 10 == 0:
                agent.load_state_dict(self.center_agent.state_dict())
            time.sleep(0.1)

        if record_performance:
            experience_queue.put(None)
            writer.close()


    def run_learner(self, agent, experience_queue):

        optimizer = torch.optim.Adam(agent.parameters(), lr=self.lr)
        while True:
            episode = experience_queue.get()
            if episode == None:
                break

            batch_state, batch_action, batch_reward, batch_next_state, _ = self.buffer.sample(self.batch_size)

            batch_state  = torch.FloatTensor(batch_state).to(self.device)
            batch_action = torch.LongTensor(batch_action).view(-1, 1).to(self.device)
            batch_reward = torch.FloatTensor(batch_reward).view(-1, 1).to(self.device)
            batch_next_state = torch.FloatTensor(batch_next_state).to(self.device)

            q_eval = agent(batch_state).gather(1, batch_action)
            q_next = agent(batch_next_state).detach()
            q_target = batch_reward + self.discount * q_next.max(1)[0].view(-1, 1)

            loss = F.mse_loss(q_eval, q_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            gc.collect()

        print('end > learner')

def main():

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env = env.unwrapped

    env.seed(0)
    utils.set_random_seed(0)

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    agent = APE_X(env_name, state_space, action_space)
    agent.train()

if __name__ == '__main__':
    #mp.set_start_method('spawn')
    main()
