# --- START OF FILE critic_network.py ---
import torch
from torch import nn

class Critic(nn.Module):
    def __init__(self,
                 problem_name,
                 embedding_dim,
                 hidden_dim,
                 # Các tham số cũ không dùng đến
                 n_heads=None, n_layers=None, normalization=None,
                 # Tham số mới
                 state_dim=6
                 ):
        super(Critic, self).__init__()
        
        self.state_dim = state_dim
        
        # Value network là một mạng MLP
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, input):
        # input bây giờ là state_for_critic từ Actor
        h_features = input
        
        # Đưa state qua value network
        baseline_value = self.value_network(h_features)
        
        return baseline_value.detach().squeeze(), baseline_value.squeeze()