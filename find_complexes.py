import numpy as np
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from persim import plot_diagrams

import gudhi as gd
from scipy.spatial.distance import pdist, squareform


import torch


import numpy as np
from scipy.sparse import csr_matrix

def distance_matrix_to_radius_graph(D, epsilon):
    """Construct graph from distance matrix using a fixed radius threshold."""
    mask = (D <= epsilon) & (D > 0)  # exclude self-connections
    adj = csr_matrix(mask.astype(np.int32))
    return adj


# Build sparse adjacency


import time
t = time.time()
from tqdm import tqdm

from utils.data_utils import read_data_df, read_data_matrix, impute_values, get_wishlist_matrix, get_wishlist_dict, evaluate, make_submission
import pickle

train_df, valid_df = read_data_df()
train_mat = read_data_matrix(train_df)
train_mat = impute_values(train_mat)
wishlist_mat = get_wishlist_matrix()
wishlist_dict = get_wishlist_dict()

D = torch.load('m_step_trans_prob.pt').cpu().numpy()  # Load precomputed distance matrix

print("Distance matrix loaded, shape:", D.shape)

min_probs = np.quantile(D, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # Generate probabilities from 0.0 to 1.0 in steps of 0.01

simplices = {}
for i, min_prob in enumerate(min_probs):
    all_points = np.array(set(train_df['sid']).union(set(10000 + train_df['pid'])))
    simplices[i] = []

    for _, row in tqdm(train_df.iterrows()):
        res = np.count_nonzero((D[row['sid'], :] >= min_prob) * (D[row['pid'] + 10000, :] >= min_prob))
        simplices[i].append(res)

    print((time.time() - t))
    print(f"Found a max of {max(simplices[i])} triangles for min_prob {min_prob}")

    with open("simplices.pkl", "wb") as f:
        pickle.dump(simplices, f)

for i, min_prob in enumerate(min_probs):
    all_points = np.array(set(valid_df['sid']).union(set(10000 + valid_df['pid'])))
    simplices[i] = []

    for _, row in tqdm(valid_df.iterrows()):
        res = np.count_nonzero((D[row['sid'], :] >= min_prob) * (D[row['pid'] + 10000, :] >= min_prob))
        simplices[i].append(res)

    print((time.time() - t))
    print(f"Found a max of {max(simplices[i])} triangles for min_prob {i}")

    with open("simplices_val.pkl", "wb") as f:
        pickle.dump(simplices, f)


