# Multi-Head DQN
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import datetime
import matplotlib.pyplot as plt
import copy
from itertools import count
from tensorboardX import SummaryWriter
from collections import namedtuple


env.seed(1)
torch.manual_seed(1)


BATCH_SIZE = 256
LR = 0.01
GAMMA = 0.95
EPSILON = 0.9
Q_MEMORY_CAPACITY = 40000
Q_NETWORK_ITERATION = 100
render = False


env = gym.make("CartPole-v0")
env = env.unwrapped
#env = wrap_deepmind(env, episode_life=False, clip_rewards=True, swap_axis=True, frame_stack=True, scale=True)
NUM_ACTIONS = env.action_space.n

# to work
class ConvNet(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc_q1 = nn.Linear(7 * 7 * 64, 512)
        self.fc_q2 = nn.Linear(512, num_actions)
        
        self.fc_t1 = nn.Linear(7 * 7 * 64, 512)
        self.fc_t2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        q_value = self.fc_q2(F.relu(self.fc_q1(x)))
        t_value = self.fc_t2(F.relu(self.fc_t1(x)))
        return q_value, t_value

    
class Net(nn.Module):
    def __init__(self, num_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc_q1 = nn.Linear(64, num_actions)
        self.fc_q2 = nn.Linear(64, num_actions)
        self.fc_q3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        q_value1 = self.fc_q1(x)
        q_value2 = self.fc_q2(x)
        q_value3 = self.fc_q3(x)
        return q_value1, q_value2, q_value3
    
    
class DQN():
    def __init__(self):
        
        super(DQN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval_net   = Net(num_actions=NUM_ACTIONS).to(self.device)
        self.target_net = Net(num_actions=NUM_ACTIONS).to(self.device)
        self.q_memory   = ReplayBuffer(Q_MEMORY_CAPACITY)
        self.optimizer = torch.optim.SGD(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.learn_step_counter = 0
        self.epsilon = 0.1
        
    def choose_action(self, state):
        
        if np.random.randn() <= self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_value1, q_value2, q_value3 = self.eval_net.forward(state)
            action = torch.max(q_value1 + q_value2 + q_value3, 1)[1].cpu().numpy()
            action = action[0] 
        else:
            action = np.random.randint(0, NUM_ACTIONS)
        return action


    def store_transition(self, state, action, reward, next_state, done):
        self.q_memory.add(state, action, reward, next_state, done)


    def learn(self):
        """
        Add Shrink to Q and T (to do)
        """
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter+=1
        
        self.optimizer.zero_grad()
        
        # ==========================
        # Q Head 1
        # ==========================
        batch_state, batch_action, batch_reward, batch_next_state, _ = self.q_memory.sample(BATCH_SIZE)
        batch_state  = torch.FloatTensor(batch_state).to(self.device)
        batch_action = torch.LongTensor(batch_action).view(-1, 1).to(self.device)
        batch_reward = torch.FloatTensor(batch_reward).view(-1, 1).to(self.device)
        batch_next_state = torch.FloatTensor(batch_next_state).to(self.device)
        
        
        q_eval1, q_eval2, q_eval3 = self.eval_net(batch_state)
        q_next1, q_next2, q_next3 = self.target_net(batch_next_state)
        q_next = ((q_next1 + q_next2 + q_next3)/3).detach()
        q_eval1 = q_eval1.gather(1, batch_action)
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval1, q_target)
        loss.backward()
        
        # ==========================
        # Q Head 2
        # ==========================
        batch_state, batch_action, batch_reward, batch_next_state, _ = self.q_memory.sample(BATCH_SIZE)
        batch_state  = torch.FloatTensor(batch_state).to(self.device)
        batch_action = torch.LongTensor(batch_action).view(-1, 1).to(self.device)
        batch_reward = torch.FloatTensor(batch_reward).view(-1, 1).to(self.device)
        batch_next_state = torch.FloatTensor(batch_next_state).to(self.device)
        
        
        q_eval1, q_eval2, q_eval3 = self.eval_net(batch_state)
        q_next1, q_next2, q_next3 = self.target_net(batch_next_state)
        q_next = ((q_next1 + q_next2 + q_next3)/3).detach()
        q_eval2 = q_eval2.gather(1, batch_action)
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval2, q_target)
        loss.backward()
        
        # ==========================
        # Q Head 3
        # ==========================
        batch_state, batch_action, batch_reward, batch_next_state, _ = self.q_memory.sample(BATCH_SIZE)
        batch_state  = torch.FloatTensor(batch_state).to(self.device)
        batch_action = torch.LongTensor(batch_action).view(-1, 1).to(self.device)
        batch_reward = torch.FloatTensor(batch_reward).view(-1, 1).to(self.device)
        batch_next_state = torch.FloatTensor(batch_next_state).to(self.device)
        
        
        q_eval1, q_eval2, q_eval3 = self.eval_net(batch_state)
        q_next1, q_next2, q_next3 = self.target_net(batch_next_state)
        q_next = ((q_next1 + q_next2 + q_next3)/3).detach()
        q_eval3 = q_eval3.gather(1, batch_action)
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval3, q_target)
        loss.backward()
        
        

        self.optimizer.step()
        
        

        if self.learn_step_counter == 100:
            self.epsilon = 0.3
        if self.learn_step_counter == 200:
            self.epsilon = 0.6
        if self.learn_step_counter == 300:
            self.epsilon = 0.99



def main():
    dqn = DQN()
    episodes = 500
    print("Collecting Experience....")
    writer = SummaryWriter("runs/MH-DQN_"+str(datetime.datetime.now()))
    avg_reward = 0
    for i in range(episodes):
        state = env.reset()
        ep_reward = 0
        t = 0
        for t in count():
            if render:
                env.render()
            action = dqn.choose_action(state)
            next_state, reward , done, info = env.step(action)
        
            dqn.store_transition(state, action, reward, next_state, done)
            ep_reward += reward
            t += 1
            if done or t >= 2000:
                break
            state = next_state
            
        if len(dqn.q_memory) >= BATCH_SIZE:
            dqn.learn()
        writer.add_scalar("live_time" , t, i+1)
        
    print("Finish")

if __name__ == '__main__':
    main()




