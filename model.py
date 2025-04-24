import torch
import torch.nn as nn
import torch.nn.functional as F

class TrafficNet(nn.Module):
    def __init__(self, n_vertices: int, n_timesteps: int):
        super().__init__()
        self.V = n_vertices
        self.T = n_timesteps

        # Convolution over traffic tensor (V, V, T) → treat T as channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.T, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Flattened conv output to embedding
        self.traffic_fc = nn.Sequential(
            nn.Linear(32 * self.V * self.V, 256),
            nn.ReLU()
        )

        # Next optimal path vector encoder
        self.path_fc = nn.Sequential(
            nn.Linear(self.V, 128),
            nn.ReLU()
        )

        # Binary status encoder (visited + current vertex)
        self.status_fc = nn.Sequential(
            nn.Linear(2 * self.V, 64),
            nn.ReLU()
        )

        # Combine all 3 encodings
        self.combined_fc = nn.Sequential(
            nn.Linear(256 + 128 + 64, 256),
            nn.ReLU()
        )

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
