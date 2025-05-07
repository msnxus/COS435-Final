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

    def forward(self, traffic_tensor, next_paths, visited_vertices, current_vertex_onehot):
        # traffic_tensor: [V, V, T]
        # next_paths: [V]
        # visited_vertices: [V] (binary)
        # current_vertex_onehot: [V] (one-hot)

        # CNN on traffic tensor
        x = traffic_tensor.permute(2, 0, 1).unsqueeze(0)  # [T, V, V] → [1, T, V, V]
        x = self.conv(x)                                 # [1, 32, V, V]
        x = x.view(1, -1)                                # flatten
        x = self.traffic_fc(x)                           # [1, 256]

        # MLP on next path distances
        p = self.path_fc(next_paths.unsqueeze(0))        # [1, 128]

        # MLP on binary status vectors
        s = torch.cat([visited_vertices, current_vertex_onehot], dim=0).unsqueeze(0)  # [1, 2V]
        s = self.status_fc(s)                            # [1, 64]

        # Merge and pass through final layers
        combined = torch.cat([x, p, s], dim=1)           # [1, 448]
        h = self.combined_fc(combined)                   # [1, 256]

        logits = self.policy_head(h)                     # [1, V]
        value = self.value_head(h)                       # [1, 1]

        return logits.squeeze(0), value.squeeze(0)
