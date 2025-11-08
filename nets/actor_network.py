# --- START OF FILE actor_network.py ---
from torch import nn
import torch
from nets.graph_layers import MultiHeadEncoder # Có thể không dùng
from torch.distributions import Categorical
import torch.nn.functional as F

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class Actor(nn.Module):

    def __init__(self,
                 problem_name,
                 embedding_dim,
                 hidden_dim,
                 n_layers, # Giữ lại để tương thích
                 normalization, # Giữ lại để tương thích
                 # Các tham số cũ không dùng đến
                 n_heads_actor=None, n_heads_decoder=None, v_range=None, seq_length=None,
                 # Tham số mới
                 state_dim=6,
                 num_actions=5
                 ):
        super(Actor, self).__init__()
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        # Thay thế kiến trúc cũ bằng một mạng MLP đơn giản
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
        print(self.get_parameter_number())

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def forward(self, problem, x_in, solution, exchange, do_sample=False, fixed_action=None, require_entropy=False, to_critic=False, only_critic=False):
        # solution, exchange không còn được dùng
        # x_in bây giờ là state của gói hàng [batch_size, state_dim]
        
        # State cho critic chính là state đầu vào
        state_for_critic = x_in
        if only_critic:
            return state_for_critic

        # 1. Lấy logits từ mạng policy
        action_logits = self.policy_network(x_in)
        
        # 2. Tạo phân phối xác suất
        action_probs = F.softmax(action_logits, dim=-1)
        log_likelihood = F.log_softmax(action_logits, dim=-1)

        # 3. Lấy mẫu hành động
        if fixed_action is not None:
            # fixed_action bây giờ là một tensor chứa chỉ số, vd: tensor([2])
            action = fixed_action.view(-1)
        else:
            if do_sample:
                dist = Categorical(action_probs)
                action = dist.sample()
            else: # Greedy
                action = action_probs.max(-1)[1]
 
        selected_log_likelihood = log_likelihood.gather(1, action.unsqueeze(-1)).squeeze(-1)
    
        # Chuẩn bị output cho PPO
        if require_entropy:
            dist = Categorical(action_probs)
            entropy = dist.entropy()
            out = (
                action,
                selected_log_likelihood,
                state_for_critic if to_critic else None,
                entropy
            )
        else:
            out = (
                action,
                selected_log_likelihood,
                state_for_critic if to_critic else None
            )
            
        return out