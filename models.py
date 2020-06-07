import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


hid1 = 512
hid2 = 256


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class ActorNN(nn.Module):

    def __init__(self, state_size, action_size, seed, drop_prob=0.2):
        super(ActorNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, hid1)
        self.fc2 = nn.Linear(hid1, hid2)
        self.fc3 = nn.Linear(hid2, action_size)
        
        self.drop1 = nn.Dropout(p=drop_prob)
        self.drop2 = nn.Dropout(p=drop_prob)
        
        self.reset_parameters()


    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

        
    def forward(self, state):
        x = F.relu(self.drop1(self.fc1(state)))
        x = F.relu(self.drop2(self.fc2(x)))
        return torch.tanh(self.fc3(x))
    
    
class CriticNN(nn.Module):
    def __init__(self, state_size, action_size, seed, drop_prob=0.2):
        super(CriticNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, hid1)
        self.fc2 = nn.Linear(hid1 + action_size, hid2)
        self.fc3 = nn.Linear(hid2, 1)

        self.drop1 = nn.Dropout(p=drop_prob)
        self.drop2 = nn.Dropout(p=drop_prob)
        
        self.reset_parameters()


    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)


    def forward(self, state, action):
        x = F.leaky_relu(self.drop1(self.fc1(state)))
        x = torch.cat((x, action), dim=1)
        x = F.leaky_relu(self.drop2(self.fc2(x)))
        return self.fc3(x)