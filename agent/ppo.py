# --- START OF FILE ppo.py ---
import os, warnings, torch, numpy as np, random
from tqdm import tqdm
from torch.utils.data import DataLoader
from nets.actor_network import Actor
from nets.critic_network import Critic
from utils import clip_grad_norms, torch_load_cpu, get_inner_model, move_to
from agent.utils import validate

class Memory:
    def __init__(self): self.actions, self.states, self.logprobs, self.rewards = [], [], [], []    
    def clear_memory(self): self.actions.clear(); self.states.clear(); self.logprobs.clear(); self.rewards.clear()

class PPO:
    def __init__(self, problem_name, size, opts):
        self.opts = opts
        state_dim, num_actions = size, opts.num_actions
        self.actor = Actor(problem_name, opts.embedding_dim, opts.hidden_dim, state_dim=state_dim, num_actions=num_actions)
        if not opts.eval_only:
            self.critic = Critic(problem_name, opts.embedding_dim, opts.hidden_dim, state_dim=state_dim)
            self.optimizer = torch.optim.Adam([{'params': self.actor.parameters(), 'lr': opts.lr_model}], lr=opts.lr_model)
            self.optimizer.add_param_group({'params': self.critic.parameters(), 'lr': opts.lr_critic})
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, opts.lr_decay)
        if opts.use_cuda and not opts.distributed:
            self.actor.to(opts.device); 
            if not opts.eval_only: self.critic.to(opts.device)
                
    
    def load(self, load_path):
        assert load_path is not None
        load_data = torch_load_cpu(load_path)
        
        model_actor = get_inner_model(self.actor)
        model_actor.load_state_dict({**load_data.get('actor', {})})
        
        if not self.opts.eval_only:
            model_critic = get_inner_model(self.critic)
            model_critic.load_state_dict({**load_data.get('critic', {})})
            self.optimizer.load_state_dict(load_data['optimizer'])
            torch.set_rng_state(load_data['rng_state'])
            if self.opts.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        print(' [*] Loading data from {}'.format(load_path))
        
    
    def save(self, epoch):
        print('Saving model and state...')
        torch.save(
            {
                'actor': get_inner_model(self.actor).state_dict(),
                'critic': get_inner_model(self.critic).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )
    
    
    def eval(self):
        torch.set_grad_enabled(False)
        self.actor.eval()
        if not self.opts.eval_only: self.critic.eval()
        
    def train(self):
        torch.set_grad_enabled(True)
        self.actor.train()
        if not self.opts.eval_only: self.critic.train()

    
    def rollout(self, problem, val_m, batch, do_sample=True, record=False, show_bar=False):
        batch = move_to(batch, self.opts.device)
        all_states = problem.input_feature_encoding(batch)
        bs = all_states.size(0)
        results = []
        pbar = tqdm(range(bs), disable=not show_bar, desc='Rollout')
        for i in pbar:
            current_state = all_states[i].unsqueeze(0)
            action, _, _, _ = self.actor(problem, current_state, None, None, do_sample=do_sample, require_entropy=True)
            _, reward, _, _, final_result = problem.step({'package_state': current_state}, None, action, None, None)
            results.append({"instance_id": i, "action": action.item(), "sequence": problem.action_pool[action.item()],
                            "reward": reward.item(), "route": final_result.get("path", []), "time": final_result.get("total_travel_time", 0)})
        print("\n" + "="*50 + "\nDETAILED ROUTING RESULTS (first 5)\n" + "="*50)
        for res in results[:5]:
            print(f"  - Instance {res['instance_id']}: Chose Action {res['action']} -> {res['sequence']}\n    - Final Route: {res['route']}\n    - Time: {res['time']:.2f}, Reward: {res['reward']:.4f}")
        print("="*50 + "\n")
        all_rewards = torch.tensor([r['reward'] for r in results])
        return all_rewards.mean(), torch.zeros(bs, 1), torch.zeros(bs, 1), all_rewards, None

    def start_inference(self, problem, val_dataset, tb_logger): validate(0, problem, self, val_dataset, tb_logger)
    def start_training(self, problem, val_dataset, tb_logger): train(0, problem, self, val_dataset, tb_logger)

def train(rank, problem, agent, val_dataset, tb_logger):
    opts = agent.opts
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    for epoch in range(opts.epoch_start, opts.epoch_end):
        agent.lr_scheduler.step(epoch)
        print(f"\n| {'Training epoch '+str(epoch):*^60} |")
        dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size)
        dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=False)
        pbar = tqdm(total=len(dataloader) * opts.K_epochs, disable=opts.no_progress_bar, desc=f'Epoch {epoch}')
        for batch in dataloader:
            train_batch(rank, problem, agent, epoch, 0, batch, tb_logger, opts, pbar)
        pbar.close()
        if (epoch % opts.checkpoint_epochs == 0): agent.save(epoch)
        validate(rank, problem, agent, val_dataset, tb_logger, _id=epoch)

def train_batch(rank, problem, agent, epoch, step, batch, tb_logger, opts, pbar):
    agent.train()
    states = problem.input_feature_encoding(move_to(batch, opts.device))
    batch_size = states.size(0)
    
    actions, log_probs, _, _ = agent.actor(problem, states, None, None, do_sample=True, require_entropy=True)
    rewards = torch.cat([problem.step({'package_state': states[i].unsqueeze(0)}, None, actions[i], None, None)[1] for i in range(batch_size)])
    
    old_states, old_actions, old_logprobs = states.detach(), actions.detach(), log_probs.detach()

    for _k in range(opts.K_epochs):
        _, logprobs, to_critic, _ = agent.actor(problem, old_states, None, None, fixed_action=old_actions, to_critic=True, require_entropy=True)
        _, bl_val = agent.critic(to_critic)
        
        advantages = rewards - bl_val.detach()
        ratios = torch.exp(logprobs - old_logprobs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - opts.eps_clip, 1 + opts.eps_clip) * advantages
        
        loss = -torch.min(surr1, surr2).mean() + ((bl_val - rewards) ** 2).mean()
        
        agent.optimizer.zero_grad()
        loss.backward()
        clip_grad_norms(agent.optimizer.param_groups, opts.max_grad_norm)
        agent.optimizer.step()
        if rank == 0: pbar.update(1)