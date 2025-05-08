import torch
import torch.nn as nn
import torch.nn.functional as F

class TravelNet(nn.Module):
    def __init__(self, N):
        super().__init__()
        input_dim = 2 * N + 1  # visited vertices + current vertex one-hot + current time
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)

        # PPO heads
        self.policy_head = nn.Linear(256, self.V)     # logits for each vertex
        self.value_head = nn.Linear(256, 1)           # value estimate

    def forward(self, current_vertex, destinations, current_time):
       

        logits = self.policy_head(h)                     # [1, V]
        value = self.value_head(h)                       # [1, 1]

        return logits.squeeze(0), value.squeeze(0)
