import time
import torch
import os
import random
import numpy as np
from utils.logger import log_to_screen, log_to_tb_val
import torch.distributed as dist
from torch.utils.data import DataLoader
from tensorboard_logger import Logger as TbLogger

def gather_tensor_and_concat(tensor):
    gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, tensor)
    return torch.cat(gather_t)

def validate(rank, problem, agent, val_dataset, tb_logger, distributed = False, _id = None):
            
    # Validate mode
    opts = agent.opts
    if rank==0: print(f'\nInference with x{opts.val_m} augments...', flush=True)
    
    agent.eval()
    problem.eval()
    
    if opts.eval_only:
        torch.manual_seed(opts.seed)
        np.random.seed(opts.seed)
        random.seed(opts.seed)
    
    val_dataset_obj = problem.make_dataset(
        size=opts.graph_size,
        num_samples=opts.val_size,
        filename=val_dataset
    )

    if distributed and opts.distributed:
        device = torch.device("cuda", rank)
        torch.distributed.init_process_group(backend='nccl', world_size=opts.world_size, rank = rank)
        torch.cuda.set_device(rank)
        agent.actor.to(device)
        agent.actor = torch.nn.parallel.DistributedDataParallel(agent.actor, device_ids=[rank])
        
        if not opts.no_tb and rank == 0:
            tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, 
                                                          opts.graph_size), opts.run_name))

        assert opts.val_size % opts.world_size == 0
        train_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size = opts.val_size // opts.world_size, shuffle=False,
                                    num_workers=0,
                                    pin_memory=True,
                                    sampler=train_sampler)
    else:
        val_dataloader = DataLoader(val_dataset_obj, batch_size=opts.val_size, shuffle=False,
                                   num_workers=0,
                                   pin_memory=True)
    
    s_time = time.time()
    
    # DataLoader chỉ lặp một lần vì batch_size = val_size
    for batch_id, batch in enumerate(val_dataloader):
        assert batch_id < 1
        # Hàm rollout bây giờ trả về (avg_reward, dummy, dummy, all_rewards, dummy)
        avg_reward, cost_hist, best_hist, all_rewards, rec_history = agent.rollout(
            problem,
            opts.val_m,
            batch,
            do_sample = True,
            record = False,
            show_bar = (rank==0)
        )
    
    if distributed and opts.distributed:
        dist.barrier()
        initial_cost = gather_tensor_and_concat(cost_hist[:,0].contiguous())
        time_used = gather_tensor_and_concat(torch.tensor([time.time() - s_time]).cuda())
        bv = gather_tensor_and_concat(bv.contiguous())
        costs_history = gather_tensor_and_concat(cost_hist.contiguous())
        search_history = gather_tensor_and_concat(best_hist.contiguous())
        reward = gather_tensor_and_concat(r.contiguous())
        dist.barrier()
    else:
        # Dữ liệu trả về từ rollout được gán lại cho phù hợp
        initial_cost = torch.tensor([0.0]) # Không có init cost
        time_used = torch.tensor([time.time() - s_time])
        bv = avg_reward # Best value bây giờ là avg_reward
        costs_history = cost_hist # Dummy tensor
        search_history = best_hist # Dummy tensor
        reward = all_rewards
        
    # log to screen  
    if rank == 0:
        log_to_screen(
            time_used, 
            initial_cost, 
            bv, 
            reward, 
            costs_history,
            search_history,
            batch_size = opts.val_size, 
            dataset_size = len(val_dataset_obj), 
            T = opts.T_max # T không còn dùng nhiều nhưng giữ lại để tương thích
        )
    
    # log to tb
    if(not opts.no_tb) and rank == 0:
        log_to_tb_val(
            tb_logger,
            time_used, 
            initial_cost, 
            bv, 
            reward, 
            costs_history,
            search_history,
            batch_size = opts.val_size,
            val_size =  opts.val_size,
            dataset_size = len(val_dataset_obj), 
            T = opts.T_max,
            epoch = _id
        )
        
    if distributed and opts.distributed: dist.barrier()

# --- END OF FILE agent/utils.py ---