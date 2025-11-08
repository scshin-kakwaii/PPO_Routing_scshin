# --- START OF FILE ppo.py ---

import os
import warnings
import torch
import numpy as np
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
from tensorboard_logger import Logger as TbLogger
import torch.multiprocessing as mp
import torch.distributed as dist

from utils import clip_grad_norms
from nets.actor_network import Actor
from nets.critic_network import Critic
from utils import torch_load_cpu, get_inner_model, move_to, move_to_cuda
from utils.logger import log_to_tb_train
from agent.utils import validate

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []    
        
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]

def lr_sd(epoch, opts):
    return opts.lr_decay ** epoch

class PPO:
    def __init__(self, problem_name, size, opts):
        
        self.opts = opts
        
        # 'size' bây giờ là state_dim cho bài toán routing
        state_dim = size 
        num_actions = opts.num_actions

        self.actor = Actor(
            problem_name=problem_name,
            embedding_dim=opts.embedding_dim,
            hidden_dim=opts.hidden_dim,
            n_layers=opts.n_encode_layers, # Giữ lại để tương thích, có thể không dùng trong Actor mới
            normalization=opts.normalization, # Giữ lại để tương thích
            state_dim=state_dim,
            num_actions=num_actions
        )
        
        if not opts.eval_only:
        
            self.critic = Critic(
                problem_name=problem_name,
                embedding_dim=opts.embedding_dim,
                hidden_dim=opts.hidden_dim,
                state_dim=state_dim
            )
        
            self.optimizer = torch.optim.Adam(
                [{'params': self.actor.parameters(), 'lr': opts.lr_model}] + 
                [{'params': self.critic.parameters(), 'lr': opts.lr_critic}]
            )
            
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, opts.lr_decay, last_epoch=-1,)
                
        print(f'Distributed: {opts.distributed}')
        
        if opts.use_cuda and not opts.distributed:
            self.actor.to(opts.device)
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

    
    def rollout(self, problem, val_m, batch, do_sample = False, record = False, show_bar = False):
        batch = move_to(batch, self.opts.device)
    
    # Dữ liệu từ dataset là 'coordinates', ta chuyển nó thành state
        all_states = problem.input_feature_encoding(batch)
    
        bs = all_states.size(0)
        results = [] # DANH SÁCH ĐỂ LƯU KẾT QUẢ CHI TIẾT

        pbar = tqdm(range(bs), disable=self.opts.no_progress_bar or not show_bar, desc = 'Rollout (Inference)')

        for i in pbar:
            current_state = all_states[i].unsqueeze(0)
        
            action, _, _, _ = self.actor(
                problem, current_state, None, None, 
                do_sample=do_sample, require_entropy=True
            )

        # BẮT LẤY KẾT QUẢ ROUTING TỪ HÀM STEP
            _, reward, _, _, final_result = problem.step(
                {'coordinates': batch['coordinates'][i].unsqueeze(0)}, 
                None, action, None, None
            )
        
        # LƯU KẾT QUẢ VÀO DANH SÁCH
            results.append({
                "instance_id": i,
                "selected_action": action.item(),
                "selected_sequence": problem.action_pool[action.item()],
                "reward": reward.item(),
                "final_route": final_result["path"],
                "total_time": final_result["total_time"],
                "total_distance": final_result["total_distance"]
            })

    # =========================================================
    # IN KẾT QUẢ CHI TIẾT RA MÀN HÌNH
    # =========================================================
        print("\n" + "="*50)
        print("DETAILED ROUTING RESULTS (showing first 5 instances)")
        print("="*50)
        for res in results[:5]:
            print(f"  - Instance {res['instance_id']}:")
            print(f"    - PPO Agent chose Action {res['selected_action']} -> Sequence: {res['selected_sequence']}")
            print(f"    - Final Route Path: {res['final_route']}")
            print(f"    - Total Time: {res['total_time']:.2f}, Total Distance: {res['total_distance']:.2f}")
            print(f"    - Received Reward: {res['reward']:.4f}")
            print("="*50 + "\n")

    # Vẫn trả về các giá trị cần thiết cho hàm validate
        all_rewards = torch.tensor([r['reward'] for r in results])
        avg_reward = all_rewards.mean()

    # (bv, cost_hist, best_hist, r, rec_history)
        return avg_reward, torch.zeros(bs, 1), torch.zeros(bs, 1), all_rewards, None

      
    def start_inference(self, problem, val_dataset, tb_logger):
        if self.opts.distributed:            
            mp.spawn(validate, nprocs=self.opts.world_size, args=(problem, self, val_dataset, tb_logger, True))
        else:
            validate(0, problem, self, val_dataset, tb_logger, distributed = False)
            
    def start_training(self, problem, val_dataset, tb_logger):
        if self.opts.distributed:
            mp.spawn(train, nprocs=self.opts.world_size, args=(problem, self, val_dataset, tb_logger))
        else:
            train(0, problem, self, val_dataset, tb_logger)
      
def train(rank, problem, agent, val_dataset, tb_logger):
    
    opts = agent.opts     
    warnings.filterwarnings("ignore")
    if opts.resume is None:
        torch.manual_seed(opts.seed)
        np.random.seed(opts.seed)
        random.seed(opts.seed)
        
    if opts.distributed:
        device = torch.device("cuda", rank)
        torch.distributed.init_process_group(backend='nccl', world_size=opts.world_size, rank = rank)
        torch.cuda.set_device(rank)
        agent.actor.to(device)
        agent.critic.to(device)
        for state in agent.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        
        agent.actor = torch.nn.parallel.DistributedDataParallel(agent.actor,
                                                               device_ids=[rank])
        if not opts.eval_only: agent.critic = torch.nn.parallel.DistributedDataParallel(agent.critic,
                                                                   device_ids=[rank])
        if not opts.no_tb and rank == 0:
            tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, 
                                                          opts.graph_size), opts.run_name))
    else:
        for state in agent.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(opts.device)
                        
    if opts.distributed: dist.barrier()
    
    # Start the actual training loop
    for epoch in range(opts.epoch_start, opts.epoch_end):
        
        # Training mode
        agent.lr_scheduler.step(epoch)
        if rank == 0:
            print('\n\n')
            print("|",format(f" Training epoch {epoch} ","*^60"),"|")
            print("Training with actor lr={:.3e} critic lr={:.3e} for run {}".format(agent.optimizer.param_groups[0]['lr'], 
                                                                                 agent.optimizer.param_groups[1]['lr'], opts.run_name) , flush=True)
        # prepare training data
        training_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size)
        if opts.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset, shuffle=False)
            training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size // opts.world_size, shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            sampler=train_sampler)
        else:
            training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, shuffle=False,
                                                       num_workers=0,
                                                       pin_memory=True)
            
        # start training
        step = epoch * (opts.epoch_size // opts.batch_size)  
        total_pbar_steps = (opts.epoch_size // opts.n_step) * opts.K_epochs
        pbar = tqdm(total=total_pbar_steps, disable=opts.no_progress_bar or rank!=0, desc='training')

        for batch_id, batch in enumerate(training_dataloader):
            # batch['coordinates'] là dữ liệu thực tế từ dataset
            num_instances_in_batch = batch['coordinates'].size(0)
            train_batch(rank, problem, agent, epoch, step, batch, tb_logger, opts, pbar)
            # Cập nhật step dựa trên số batch đã xử lý, không phải số instance
            step += 1 # Tăng step lên 1 sau mỗi batch
        pbar.close()
        
        # save new model after one epoch  
        if rank == 0 and not opts.distributed: 
            if not opts.no_saving and (( opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or \
                        epoch == opts.epoch_end - 1): agent.save(epoch)
        elif opts.distributed and rank == 1:
            if not opts.no_saving and (( opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or \
                        epoch == opts.epoch_end - 1): agent.save(epoch)
            
        
        # validate the new model        
        if rank == 0: validate(rank, problem, agent, val_dataset, tb_logger, _id = epoch)
        
        # syn
        if rank == 0 and not opts.distributed: 
            if not opts.no_saving and (( opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or \
                        epoch == opts.epoch_end - 1): agent.save(epoch)
        elif opts.distributed and rank == 1:
            if not opts.no_saving and (( opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or \
                        epoch == opts.epoch_end - 1): agent.save(epoch)
            
        if rank == 0: validate(rank, problem, agent, val_dataset, tb_logger, _id = epoch)
        
        if opts.distributed: dist.barrier()

    
def train_batch(rank, problem, agent, epoch, step, batch, tb_logger, opts, pbar):
    agent.train()
    problem.train()
    
    batch = move_to(batch, opts.device)
    # Dữ liệu thực tế từ dataloader là `batch['coordinates']`
    # Ta chuyển nó thành `states` để đưa vào Actor
    states = problem.input_feature_encoding(batch)
    batch_size = states.size(0) # Số lượng instance trong batch
    
    # THU THẬP KINH NGHIỆM TỪ TOÀN BỘ BATCH
    actions, log_probs, _to_critic, entropies = agent.actor(
        problem, states, None, None, do_sample=True, require_entropy=True, to_critic=True
    )
    
    rewards = []
    # Lặp qua từng instance trong batch để gọi step
    for i in range(batch_size):
        instance_batch = {'coordinates': batch['coordinates'][i].unsqueeze(0)}
        
        # Gọi step và nhận 5 giá trị, nhưng chỉ dùng reward (r)
        _, r, _, _, _ = problem.step(instance_batch, None, actions[i], None, None)
        
        rewards.append(r)
    rewards = torch.cat(rewards)

    # CẬP NHẬT PPO (Phần này giữ nguyên từ trước)
    old_states = states.detach()
    old_actions = actions.detach()
    old_logprobs = log_probs.detach()
    
    for _k in range(opts.K_epochs):
        _, logprobs, to_critic, _ = agent.actor(
            problem, old_states, None, None, fixed_action=old_actions, to_critic=True, require_entropy=True
        )
        _, bl_val = agent.critic(to_critic)
        bl_val_detached = bl_val.detach()
        Reward = rewards
        advantages = Reward - bl_val_detached
        ratios = torch.exp(logprobs - old_logprobs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - opts.eps_clip, 1 + opts.eps_clip) * advantages
        reinforce_loss = -torch.min(surr1, surr2).mean()
        baseline_loss = ((bl_val - Reward) ** 2).mean()
        loss = baseline_loss + reinforce_loss
        agent.optimizer.zero_grad()
        loss.backward()
        clip_grad_norms(agent.optimizer.param_groups, opts.max_grad_norm)
        agent.optimizer.step()
        if rank == 0: pbar.update(1)

# --- END OF FILE ppo.py ---