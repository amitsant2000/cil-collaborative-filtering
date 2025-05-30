import pandas as pd
import numpy as np
from collections import defaultdict
import heapq
import torch

import torch.nn.functional as F

from tqdm import tqdm

from utils.data_utils import read_data_df, read_data_matrix, impute_values, get_wishlist_matrix, get_wishlist_dict, evaluate, make_submission

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DISCOUNT_FACTOR = 0.99

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df, valid_df = read_data_df()

transition_probability = np.eye((10000 + 1000), dtype=np.float32)

# === Build bipartite graph ===
graph = defaultdict(list)
for _, row in train_df.iterrows():
    sid = f"s{row['sid']}"
    pid = f"p{row['pid']}"
    weight = (row["rating"]) / 5  # Inverse weight
    transition_probability[row['sid'], row['pid'] + 10_000] = weight
    transition_probability[row['pid'] + 10_000, row['sid']] = weight

trainsition_probability = torch.from_numpy(transition_probability).to(device)
transition_probability = F.normalize(trainsition_probability, p=1, dim=1)

m = 3

m_step_transition_probability = torch.eye((10000 + 1000), device=device)

for _ in range(m):
    m_step_transition_probability = torch.matmul(transition_probability.T, m_step_transition_probability)

# Save or pass to model
torch.save(m_step_transition_probability, f"{m}_step_trans_prob.pt")

# sid_indices = train_df['sid'].values
# pid_indices = train_df['pid'].values
# distance_indices = (sid_indices, 10000 + pid_indices)
# # Compute the difference vectors for each (sid, pid) pair in batches to avoid memory issues
# batch_size = 100_000
# num_samples = len(distance_indices[0])
# distances_list = []

# for start in tqdm(range(0, num_samples, batch_size)):
#     end = min(start + batch_size, num_samples)
#     sids = distance_indices[0][start:end]
#     pids = distance_indices[1][start:end]
#     diffs = m_step_transition_probability[sids] - m_step_transition_probability[pids]
#     batch_distances = torch.norm(diffs, dim=1).cpu()
#     distances_list.append(batch_distances)

# distances = torch.concatenate(distances_list)


# torch.save(distances, "diffusion_distances.pt")