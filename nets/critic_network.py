import torch
from torch import nn

class Critic(nn.Module):
    def __init__(self, problem_name, embedding_dim, hidden_dim, state_dim=6, **kwargs):
        super(Critic, self).__init__()
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, input):
        baseline_value = self.value_network(input)
        return baseline_value.detach().squeeze(-1), baseline_value.squeeze(-1)