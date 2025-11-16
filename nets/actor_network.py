from torch import nn
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, problem_name, embedding_dim, hidden_dim, state_dim=6, num_actions=50, **kwargs):
        super(Actor, self).__init__()
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    def forward(self, problem, x_in, solution, exchange, do_sample=False, fixed_action=None, require_entropy=False, to_critic=False, only_critic=False):
        if only_critic: return x_in
        action_logits = self.policy_network(x_in)
        action_probs = F.softmax(action_logits, dim=-1)
        log_likelihood = F.log_softmax(action_logits, dim=-1)
        if fixed_action is not None: action = fixed_action.view(-1)
        else: action = action_probs.multinomial(1).squeeze(1) if do_sample else action_probs.max(-1)[1]
        selected_log_likelihood = log_likelihood.gather(1, action.unsqueeze(-1)).squeeze(-1)
        out = (action, selected_log_likelihood, x_in if to_critic else None)
        if require_entropy:
            entropy = Categorical(action_probs).entropy()
            out = out + (entropy,)
        return out