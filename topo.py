import pandas as pd
import numpy as np
from collections import defaultdict
import heapq
import torch

from tqdm import tqdm

from utils.data_utils import read_data_df, read_data_matrix, impute_values, get_wishlist_matrix, get_wishlist_dict, evaluate, make_submission


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

MAX_NODES = 5000  # Maximum number of nodes to process in Dijkstra's algorithm
DISCOUNT_FACTOR = 0.99

train_df, valid_df = read_data_df()
train_mat = read_data_matrix(train_df)
train_mat = impute_values(train_mat)
wishlist_mat = get_wishlist_matrix()
wishlist_dict = get_wishlist_dict()

# === Build bipartite graph ===
graph = defaultdict(list)
for _, row in train_df.iterrows():
    sid = f"s{row['sid']}"
    pid = f"p{row['pid']}"
    weight = (row["rating"]) / 5  # Inverse weight
    graph[sid].append((pid, weight))
    graph[pid].append((sid, weight))

# === Dijkstra's algorithm ===
def dijkstra(graph, source, max_nodes=10000):
    dist = {}
    visited = set()
    queue = [(-1, source)]

    while queue and len(dist) < max_nodes:
        d, node = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        dist[node] = -d
        for neighbor, w in graph[node]:
            if neighbor not in visited:
                heapq.heappush(queue, (DISCOUNT_FACTOR * d * w, neighbor))
    return dist

# === Generate distance matrix ===
sids = [i for i in range(10000)]  # Assuming 10,000 scientists
pids = [i for i in range(1000)]  # Assuming 1,000 papers

sid_to_idx = {sid: i for i, sid in enumerate(sids)}
pid_to_idx = {pid: 10_000 + i for i, pid in enumerate(pids)}
distance_matrix = np.full((len(sids) + len(pids), len(sids) + len(pids)), fill_value=0.0, dtype=np.float32)

import concurrent.futures

def process_sid(sid):
    source = f"s{sid}"
    distances = dijkstra(graph, source, max_nodes=MAX_NODES)
    updates = []
    for target_node, d in distances.items():
        if target_node.startswith("p"):
            pid = int(target_node[1:])
            if pid in pid_to_idx:
                updates.append((sid_to_idx[sid], pid_to_idx[pid], d))
                updates.append((pid_to_idx[pid], sid_to_idx[sid], d))
        else:
            target_sid = int(target_node[1:])
            if target_sid in sid_to_idx:
                updates.append((sid_to_idx[sid], sid_to_idx[target_sid], d))
                updates.append((sid_to_idx[target_sid], sid_to_idx[sid], d))
    return updates

def process_pid(pid):
    source = f"p{pid}"
    distances = dijkstra(graph, source, max_nodes=MAX_NODES)
    updates = []
    for target_node, d in distances.items():
        if target_node.startswith("s"):
            sid = int(target_node[1:])
            if sid in sid_to_idx:
                updates.append((pid_to_idx[pid], sid_to_idx[sid], d))
                updates.append((sid_to_idx[sid], pid_to_idx[pid], d))
        else:
            target_pid = int(target_node[1:])
            if target_pid in pid_to_idx:
                updates.append((pid_to_idx[pid], pid_to_idx[target_pid], d))
                updates.append((pid_to_idx[target_pid], pid_to_idx[pid], d))
    return updates

with concurrent.futures.ThreadPoolExecutor(30) as executor:
    # Process sids in parallel
    sid_futures = [executor.submit(process_sid, sid) for sid in sids]
    for future in tqdm(concurrent.futures.as_completed(sid_futures), total=len(sids)):
        for i, j, d in future.result():
            distance_matrix[i, j] = d

    # Process pids in parallel
    pid_futures = [executor.submit(process_pid, pid) for pid in pids]
    for future in tqdm(concurrent.futures.as_completed(pid_futures), total=len(pids)):
        for i, j, d in future.result():
            distance_matrix[i, j] = d

# === Normalize and convert to tensor ===
# distance_matrix = np.nan_to_num(distance_matrix, nan=np.inf, posinf=np.max(distance_matrix[distance_matrix < np.inf]) + 1.0)
distance_tensor = torch.from_numpy(distance_matrix)


print(((distance_matrix < 1) * (distance_matrix > 0)).any())

print(((distance_tensor < 1) * (distance_tensor > 0)).any())

# Save or pass to model
torch.save(distance_tensor, "shortest_path_distances_1000.pt")
