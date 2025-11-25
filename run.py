# --- START OF FILE run.py (phiên bản đã sửa lỗi lưu file) ---
import os, json, torch, pprint, numpy as np, warnings, random
from options import get_options
from problems.problem_routing import ROUTING
from agent.ppo import PPO
from agent.utils import validate
from tensorboard_logger import Logger as TbLogger

def load_agent(name): return {'ppo': PPO}.get(name, None)
def load_problem(name): return {'routing': ROUTING}.get(name, None)

def run(opts):
    pprint.pprint(vars(opts))
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    # ================== CẤU HÌNH CHO BÀI TOÁN ==================
    opts.problem = 'routing'
    opts.graph_size = 50 # Số lượng bus stops
    opts.epoch_size = 1280
    opts.batch_size = 128
    opts.val_dataset = None
    # ==========================================================
    
    # TẠO ĐƯỜNG DẪN LƯU TRỮ DỰA TRÊN OPTS ĐÃ CẬP NHẬT
    opts.save_dir = os.path.join(
        opts.output_dir,
        f"{opts.problem}_{opts.graph_size}",
        opts.run_name
    )

    # KIỂM TRA VÀ TẠO THƯ MỤC NẾU CHƯA TỒN TẠI (ĐÂY LÀ PHẦN SỬA LỖI)
    if not opts.no_saving and not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
        print(f"Created save directory: {opts.save_dir}")

    # Cấu hình Tensorboard
    tb_logger = TbLogger(os.path.join(opts.log_dir, f"{opts.problem}_{opts.graph_size}", opts.run_name)) if not opts.no_tb else None

    # Lưu lại file cấu hình
    if not opts.no_saving:
        with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)

    opts.device = torch.device("cuda" if opts.use_cuda and torch.cuda.is_available() else "cpu")
    
    problem = load_problem(opts.problem)(p_size=opts.graph_size)
    opts.num_actions = problem.num_actions

    agent = load_agent(opts.RL_agent)(problem.NAME, problem.state_dim, opts)

    if opts.load_path: agent.load(opts.load_path)
    
    if opts.eval_only:
        validate(0, problem, agent, opts.val_dataset, tb_logger)
    else:
        agent.start_training(problem, opts.val_dataset, tb_logger)            

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    run(get_options())
# --- END OF FILE run.py ---