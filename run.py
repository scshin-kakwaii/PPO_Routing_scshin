# --- START OF FILE run.py (phiên bản tối giản để chạy) ---
import os
import json
import torch
import pprint
import numpy as np
from tensorboard_logger import Logger as TbLogger
import warnings
import random
from options import get_options

from problems.problem_tsp import TSP
from problems.problem_vrp import CVRP
from problems.problem_routing_simple import ROUTING_SIMPLE # THAY ĐỔI Ở ĐÂY
from agent.ppo import PPO

def load_agent(name):
    agent = {'ppo': PPO}.get(name, None)
    assert agent is not None, "Currently unsupported agent: {}!".format(name)
    return agent

def load_problem(name):
    problem = {
        'tsp': TSP,
        'vrp': CVRP,
        'routing_simple': ROUTING_SIMPLE, # THAY ĐỔI Ở ĐÂY
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem

def run(opts):
    pprint.pprint(vars(opts))
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    # ============== CẤU HÌNH CHO BÀI TOÁN TỐI GIẢN ==============
    opts.problem = 'routing_simple' # THAY ĐỔI Ở ĐÂY
    opts.graph_size = 20  # Vẫn dùng để tạo TSPDataset, nhưng chỉ 3 nút đầu được dùng
    opts.num_actions = 5
    
    # Các tham số này nên được đặt trong options.py, nhưng bạn có thể ghi đè ở đây để test
    opts.epoch_size = 1280
    opts.batch_size = 128
    opts.val_dataset = None
    # ==========================================================

    tb_logger = None
    if not opts.no_tb:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    if not opts.no_saving and not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
        
    if not opts.no_saving:
        with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)

    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
    
    problem = load_problem(opts.problem)(p_size=opts.graph_size)
    opts.num_actions = problem.num_actions

    # state_dim được lấy từ problem, sau đó truyền vào agent thông qua `problem.state_dim`
    agent = load_agent(opts.RL_agent)(problem.NAME, problem.state_dim, opts)

    if opts.load_path is not None:
        agent.load(opts.load_path)
    
    if opts.eval_only:
        agent.start_inference(problem, opts.val_dataset, tb_logger)
    else:
        agent.start_training(problem, opts.val_dataset, tb_logger)            

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    run(get_options())
# --- END OF FILE run.py ---