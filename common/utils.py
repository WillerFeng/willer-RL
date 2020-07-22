import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def reward_shape(origin_reward, discount):
    length = len(origin_reward)
    new_reward = np.zeros_like(origin_reward, dtype=float)
    for i in reversed(range(length)):
        new_reward[i] = origin_reward[i] + (discount * new_reward[i+1] if i+1 < length else 0)
    return new_reward