# --- START OF FILE problem_routing.py ---
from torch.utils.data import Dataset
import torch
import numpy as np
import os
import pickle
import random
import itertools
import networkx as nx
import heapq

# ==============================================================================
# Lớp mô phỏng Môi trường Đồ thị (Graph Environment)
# ==============================================================================
class GraphEnvironment:
    def __init__(self, num_stops=20):
        self.num_stops = num_stops
        self.graph = nx.DiGraph()
        self._generate_bus_network()

    def _generate_bus_network(self):
        for i in range(self.num_stops):
            self.graph.add_node(i, pos=(random.uniform(0, 100), random.uniform(0, 100)))
        for i in range(self.num_stops):
            num_connections = random.randint(2, 5)
            for _ in range(num_connections):
                j = random.randint(0, self.num_stops - 1)
                if i != j and not self.graph.has_edge(i, j):
                    distance = np.linalg.norm(np.array(self.graph.nodes[i]['pos']) - np.array(self.graph.nodes[j]['pos']))
                    travel_time = distance / 20.0
                    self.graph.add_edge(i, j, travel_time=travel_time, distance=distance)
        print(f"Generated a bus network with {self.graph.number_of_nodes()} stops and {self.graph.number_of_edges()} routes.")

    def run_dijkstra(self, origin_stop, dest_stop, **kwargs):
        print("  >> Running Dijkstra...")
        try:
            travel_time, path = nx.single_source_dijkstra(self.graph, source=origin_stop, target=dest_stop, weight='travel_time')
            total_distance = sum(self.graph.edges[path[i], path[i+1]]['distance'] for i in range(len(path) - 1))
            return {"total_travel_time": travel_time, "total_distance": total_distance, "path": path}
        except nx.NetworkXNoPath:
            return {"total_travel_time": float('inf'), "total_distance": float('inf'), "path": []}

    def run_a_star(self, origin_stop, dest_stop, **kwargs):
        print("  >> Running A* Search...")
        def heuristic(u, v):
            pos1 = self.graph.nodes[u]['pos']; pos2 = self.graph.nodes[v]['pos']
            return np.linalg.norm(np.array(pos1) - np.array(pos2)) / 25.0
        try:
            path = nx.astar_path(self.graph, source=origin_stop, target=dest_stop, heuristic=heuristic, weight='travel_time')
            total_travel_time = sum(self.graph.edges[path[i], path[i+1]]['travel_time'] for i in range(len(path) - 1))
            total_distance = sum(self.graph.edges[path[i], path[i+1]]['distance'] for i in range(len(path) - 1))
            return {"total_travel_time": total_travel_time, "total_distance": total_distance, "path": path}
        except nx.NetworkXNoPath:
            return {"total_travel_time": float('inf'), "total_distance": float('inf'), "path": []}
    
    # TODO: Thêm các hàm run_ga, run_sa, run_ts thực tế ở đây
    def run_ga(self, origin_stop, dest_stop, **kwargs): return self.run_dijkstra(origin_stop, dest_stop)
    def run_sa(self, origin_stop, dest_stop, **kwargs): return self.run_a_star(origin_stop, dest_stop)
    def run_ts(self, origin_stop, dest_stop, **kwargs): return self.run_dijkstra(origin_stop, dest_stop)


class RoutingDataset(Dataset):
    def __init__(self, num_samples=10000, num_stops=20, **kwargs):
        super(RoutingDataset, self).__init__()
        self.data = []
        self.state_dim = 6
        for i in range(num_samples):
            origin_stop = random.randint(0, num_stops - 1)
            dest_stop = random.randint(0, num_stops - 1)
            while dest_stop == origin_stop: dest_stop = random.randint(0, num_stops - 1)
            start_time = np.random.uniform(0, 3600 * 8)
            deadline = start_time + np.random.uniform(1800, 3600 * 2)
            state = torch.FloatTensor([origin_stop, dest_stop, 0, 0, start_time, deadline])
            self.data.append({'package_state': state})
        self.N = len(self.data)
        print(f'{self.N} package instances initialized for a network of {num_stops} stops.')
    def __len__(self): return self.N
    def __getitem__(self, idx): return self.data[idx]

class ROUTING(object):
    NAME = 'routing'
    
    def __init__(self, p_size, with_assert=False, **kwargs):
        self.num_stops = p_size
        self.graph_env = GraphEnvironment(num_stops=self.num_stops)
        self.state_dim = 6

        self.algorithm_functions = {
            "dijkstra": self.graph_env.run_dijkstra, "a_star": self.graph_env.run_a_star,
            "ga": self.graph_env.run_ga, "sa": self.graph_env.run_sa, "ts": self.graph_env.run_ts
        }
        
        self.action_pool = self._generate_algorithm_sequences(num_sequences=50, max_len=3)
        self.num_actions = len(self.action_pool)
        print(f'Routing problem with {self.num_actions} auto-generated ALGORITHM SEQUENCES.')
        print("Sample sequences:", self.action_pool[:3])
        self.train()

    def _generate_algorithm_sequences(self, num_sequences, max_len):
        sequences = set()
        algo_names = list(self.algorithm_functions.keys())
        while len(sequences) < num_sequences:
            seq_len = random.randint(1, max_len)
            sequence = tuple(random.choices(algo_names, k=seq_len))
            sequences.add(sequence)
        return [list(seq) for seq in sorted(list(sequences))]

    def eval(self, perturb=False): self.training = False
    def train(self): self.training = True
    def input_feature_encoding(self, batch): return batch['package_state']
    def get_initial_solutions(self, batch): return None

    def _calculate_reward(self, route_result, package_state):
        w_d, w_t, w_feasibility, w_penalty = 0.4, 0.6, 0.3, 0.5
        MAX_PRACTICAL_DISTANCE, d_max, t_max = 500, 1000, 7200
        start_time, deadline = package_state[4], package_state[5]
        max_allowed_time = deadline - start_time
        
        if route_result["total_travel_time"] == float('inf'): return torch.tensor([-10.0])
        
        travel_time = route_result["total_travel_time"]
        num_hops = len(route_result.get("path", [])) - 1
        waiting_time = (num_hops - 1) * 300 if num_hops > 1 else 0
        total_time = travel_time + waiting_time
        
        d_hat = route_result["total_distance"] / d_max
        t_hat = total_time / t_max
        base_reward = -(w_d * d_hat + w_t * t_hat)

        if total_time <= max_allowed_time:
            feasibility_bonus = w_feasibility * (1.0 - (total_time / max_allowed_time))
        else:
            feasibility_bonus = -w_feasibility * (total_time / max_allowed_time)
        
        distance_penalty = 0
        if route_result["total_distance"] > MAX_PRACTICAL_DISTANCE:
            distance_penalty = w_penalty * (route_result["total_distance"] / MAX_PRACTICAL_DISTANCE - 1.0)
            
        total_reward = base_reward + feasibility_bonus - distance_penalty
        return torch.tensor([total_reward], dtype=torch.float32)

    def step(self, batch, rec, action, obj, solving_state):
        action_idx = action.item()
        selected_sequence = self.action_pool[action_idx]
        package_state_tensor = batch['package_state'][0]
        package_state_numpy = package_state_tensor.cpu().numpy()
        origin_stop, dest_stop = int(package_state_numpy[0]), int(package_state_numpy[1])
        
        final_result = None
        current_solution_info = {}
        for algo_name in selected_sequence:
            algo_function = self.algorithm_functions[algo_name]
            final_result = algo_function(origin_stop, dest_stop, initial_solution=current_solution_info)
            current_solution_info = final_result
        
        reward = self._calculate_reward(final_result, package_state_numpy).to(action.device)
        new_obj = -reward.unsqueeze(0)
        return None, reward, new_obj, None, final_result
        
    @staticmethod
    def make_dataset(*args, **kwargs):
        return RoutingDataset(num_stops=kwargs.get('size', 20), num_samples=kwargs.get('num_samples', 10000))