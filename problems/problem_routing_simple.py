# --- START OF FILE problem_routing_simple.py (phiên bản mới hỗ trợ SEQUENCE) ---
from torch.utils.data import Dataset
import torch
import numpy as np
import os
import pickle

# ==============================================================================
# PHẦN GIẢ LẬP - CÁC HÀM CHẠY THUẬT TOÁN ĐƠN LẺ
# Chúng ta sẽ sửa đổi chúng một chút để có thể nhận input là một giải pháp có sẵn
# ==============================================================================
def run_dijkstra(package_state, initial_solution=None):
    print("  >> Running Dijkstra...")
    # Logic thực tế có thể không cần initial_solution
    path = [0, np.random.randint(1, 10), 10]
    return {
        "total_time": np.random.uniform(110, 150), 
        "total_distance": np.random.uniform(4800, 5500), 
        "path": path # TRẢ VỀ ĐƯỜNG ĐI
    }

def run_a_star(package_state, initial_solution=None):
    print("  >> Running A* Search...")
    path = [0, np.random.randint(1, 10), np.random.randint(11, 20), 20]
    return {
        "total_time": np.random.uniform(100, 140), 
        "total_distance": np.random.uniform(4500, 5200), 
        "path": path # TRẢ VỀ ĐƯỜNG ĐI
    }

def run_ga(package_state, initial_solution=None):
    print("  >> Running Genetic Algorithm...")
    path = initial_solution["path"] + [np.random.randint(21, 30)] if initial_solution else [0, 9, 19, 29]
    return {
        "total_time": np.random.uniform(120, 180), 
        "total_distance": np.random.uniform(5000, 6000), 
        "path": path, # TRẢ VỀ ĐƯỜNG ĐI
        "solution_object": {"path": path} # Đảm bảo solution_object chứa path
    }
def run_sa(package_state, initial_solution=None):
    print("  >> Running Simulated Annealing...")
    path = initial_solution["path"] + [np.random.randint(31, 40)] if initial_solution else [0, 8, 18, 28, 38]
    return {
        "total_time": np.random.uniform(115, 170), 
        "total_distance": np.random.uniform(4900, 5800), 
        "path": path, # TRẢ VỀ ĐƯỜNG ĐI
        "solution_object": {"path": path}
    }
    
def run_ts(package_state, initial_solution=None):
    print("  >> Running Tabu Search...")
    path = initial_solution["path"] + [np.random.randint(41, 50)] if initial_solution else [0, 7, 17, 27, 37, 47]
    return {
        "total_time": np.random.uniform(105, 160), 
        "total_distance": np.random.uniform(4700, 5600), 
        "path": path, # TRẢ VỀ ĐƯỜNG ĐI
        "solution_object": {"path": path}
    }
# ==============================================================================

# Giữ lại TSPDataset từ phiên bản trước để code chạy được ngay
class TSPDataset(Dataset):
    def __init__(self, filename=None, size=20, num_samples=10000, offset=0, **kwargs):
        super(TSPDataset, self).__init__()
        self.data = []
        if filename is not None and os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [ {'coordinates': torch.FloatTensor(d)} for d in data[offset:offset+num_samples] ]
        else:
            self.data = [{'coordinates': torch.FloatTensor(size, 2).uniform_(0, 1)} for _ in range(num_samples)]
        self.size = len(self.data)
        print(f'{self.size} instances from TSPDataset initialized for simple routing simulation.')
    def __len__(self): return self.size
    def __getitem__(self, idx): return self.data[idx]


class ROUTING_SIMPLE(object):
    NAME = 'routing_simple'
    
    def __init__(self, p_size, with_assert=False, **kwargs):
        
        self.algorithm_functions = {
            "dijkstra": run_dijkstra,
            "a_star": run_a_star,
            "ga": run_ga,
            "sa": run_sa,
            "ts": run_ts
        }
        
        # =========================================================
        # ĐÂY LÀ ACTION POOL CỦA BẠN
        # =========================================================
        self.action_pool = [
            ["ga", "ts"],                 # Action 0
            ["sa", "ts"],                 # Action 1
            ["dijkstra", "a_star", "ga"], # Action 2
            ["a_star", "sa"],             # Action 3
            ["ts", "ga", "sa"]            # Action 4
        ]
        
        self.num_actions = len(self.action_pool)
        self.state_dim = 6
        print(f'Simple Routing problem with {self.num_actions} ALGORITHM SEQUENCES.')
        self.train()

    def eval(self, perturb=False): self.training = False
    def train(self): self.training = True
    def input_feature_encoding(self, batch):
        coords = batch['coordinates']
        batch_size = coords.size(0)
        simple_state = coords[:, :3, :].reshape(batch_size, -1)
        if simple_state.size(1) < self.state_dim:
            padding = torch.zeros(batch_size, self.state_dim - simple_state.size(1), device=coords.device)
            simple_state = torch.cat([simple_state, padding], dim=1)
        return simple_state
    def get_initial_solutions(self, batch): return None

    # Hàm reward đơn giản vẫn có thể dùng
    def _calculate_simple_reward(self, final_result):
        # Bây giờ reward có thể dựa vào kết quả cuối cùng
        # Ví dụ: reward càng cao nếu total_time càng thấp
        reward = 1.0 / (final_result["total_time"] / 100.0) # Reward tỉ lệ nghịch với thời gian
        return torch.tensor([reward])

    # =========================================================
    # THAY ĐỔI LỚN TRONG HÀM STEP
    # =========================================================
    def step(self, batch, rec, action, obj, solving_state):
        action_idx = action.item()
        selected_sequence = self.action_pool[action_idx]
        
        print(f"Executing sequence {action_idx}: {selected_sequence}")

        dummy_package_state = {} 
        current_solution = None
        final_result = None
        
        for algo_name in selected_sequence:
            algo_function = self.algorithm_functions[algo_name]
            result = algo_function(dummy_package_state, initial_solution=current_solution)
            # Truyền `path` vào `solution_object` để bước sau có thể dùng
            current_solution = {"path": result["path"]}
            final_result = result
        
        reward = self._calculate_simple_reward(final_result).to(action.device)
        
        next_state = rec 
        new_obj = -reward.unsqueeze(0)
        
        # TRẢ VỀ final_result ĐỂ CÁC HÀM KHÁC CÓ THỂ ĐỌC
        return next_state, reward, new_obj, solving_state, final_result
        
    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)