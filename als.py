import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from utils.data_utils import read_data_df, read_data_matrix, get_wishlist_matrix, get_wishlist_dict, impute_values

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


train_df, valid_df = read_data_df()

def remap_ids(df):
    sid_map = {sid: i for i, sid in enumerate(df['sid'].unique())}
    pid_map = {pid: i for i, pid in enumerate(df['pid'].unique())}
    df = df.copy()
    df['sid'] = df['sid'].map(sid_map)
    df['pid'] = df['pid'].map(pid_map)
    return df, sid_map, pid_map

def als_explicit(train_df, num_factors=32, num_iters=10, reg=0.1):
    num_users = train_df['sid'].max() + 1
    num_items = train_df['pid'].max() + 1

    U = torch.randn(num_users, num_factors)
    V = torch.randn(num_items, num_factors)

    for _ in range(num_iters):
        # Fix V, solve for U
        for uid in range(num_users):
            idx = train_df[train_df['sid'] == uid].index
            items = train_df.loc[idx, 'pid'].values
            ratings = torch.tensor(train_df.loc[idx, 'rating'].values, dtype=torch.float32)
            V_i = V[items]
            A = V_i.T @ V_i + reg * torch.eye(num_factors)
            b = V_i.T @ ratings
            U[uid] = torch.linalg.solve(A, b)

        # Fix U, solve for V
        for iid in range(num_items):
            idx = train_df[train_df['pid'] == iid].index
            users = train_df.loc[idx, 'sid'].values
            ratings = torch.tensor(train_df.loc[idx, 'rating'].values, dtype=torch.float32)
            U_i = U[users]
            A = U_i.T @ U_i + reg * torch.eye(num_factors)
            b = U_i.T @ ratings
            V[iid] = torch.linalg.solve(A, b)

    return U, V


def predict_als(U, V, df):
    preds = []
    for _, row in df.iterrows():
        u, v = row['sid'], row['pid']
        if u >= len(U) or v >= len(V):
            preds.append(3.0)  # fallback
        else:
            preds.append(U[u] @ V[v])
    return np.clip(preds, 1, 5)

# Step 0: remap IDs to indices
train_df, sid_map, pid_map = remap_ids(train_df)
valid_df = valid_df[valid_df['sid'].isin(sid_map) & valid_df['pid'].isin(pid_map)].copy()
valid_df['sid'] = valid_df['sid'].map(sid_map)
valid_df['pid'] = valid_df['pid'].map(pid_map)

# Step 1: train ALS
U, V = als_explicit(train_df, num_factors=64, num_iters=15, reg=0.05)

# Step 2: evaluate
preds = predict_als(U, V, valid_df)
rmse = root_mean_squared_error(valid_df['rating'], preds)
print(f"Validation RMSE: {rmse:.4f}")
