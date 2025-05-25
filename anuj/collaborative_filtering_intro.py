
# Collaborative filtering project

from typing import Tuple, Callable

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import os
from tqdm import tqdm, trange


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

## Helper functions
DATA_DIR = "."


def read_data_df() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reads in data and splits it into training and validation sets with a 75/25 split."""
    
    df = pd.read_csv(os.path.join(DATA_DIR, "train_ratings.csv"))

    # Split sid_pid into sid and pid columns
    df[["sid", "pid"]] = df["sid_pid"].str.split("_", expand=True)
    df = df.drop("sid_pid", axis=1)
    df["sid"] = df["sid"].astype(int)
    df["pid"] = df["pid"].astype(int)
    
    # Split into train and validation dataset
    train_df, valid_df = train_test_split(df, test_size=0.1)
    return train_df, valid_df


def read_data_matrix(df: pd.DataFrame) -> np.ndarray:
    """Returns matrix view of the training data, where columns are scientists (sid) and
    rows are papers (pid)."""

    return df.pivot(index="sid", columns="pid", values="rating").values


def evaluate(valid_df: pd.DataFrame, pred_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> float:
    """
    Inputs:
        valid_df: Validation data, returned from read_data_df for example.
        pred_fn: Function that takes in arrays of sid and pid and outputs their rating predictions.

    Outputs: Validation RMSE
    """
    
    preds = pred_fn(valid_df["sid"].values, valid_df["pid"].values)
    return root_mean_squared_error(valid_df["rating"].values, preds)


def make_submission(pred_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], filename: os.PathLike):
    """Makes a submission CSV file that can be submitted to kaggle.

    Inputs:
        pred_fn: Function that takes in arrays of sid and pid and outputs a score.
        filename: File to save the submission to.
    """
    
    df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

    # Get sids and pids
    sid_pid = df["sid_pid"].str.split("_", expand=True)
    sids = sid_pid[0]
    pids = sid_pid[1]
    sids = sids.astype(int).values
    pids = pids.astype(int).values
    
    df["rating"] = pred_fn(sids, pids)
    print('saving to:', filename)
    df.to_csv(filename, index=False)

## Singular value decomposition

def impute_values(mat: np.ndarray) -> np.ndarray:
    return np.nan_to_num(mat, nan=3.0)

def opt_rank_k_approximation(m: np.ndarray, k: int):
    """Returns the optimal rank-k reconstruction matrix, using SVD."""
    
    assert 0 < k <= np.min(m.shape), f"The rank must be in [0, min(m, n)]"
    
    U, S, Vh = np.linalg.svd(m, full_matrices=False)
    
    U_k = U[:, :k]
    S_k = S[:k]
    Vh_k = Vh[:k]
    
    return np.dot(U_k * S_k, Vh_k)


def matrix_pred_fn(train_recon: np.ndarray, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
    """
    Input:
        train_recon: (M, N) matrix with predicted values for every (sid, pid) pair.
        sids: (D,) vector with integer scientist IDs.
        pids: (D,) vector with integer paper IDs.
        
    Outputs: (D,) vector with predictions.
    """
    
    return train_recon[sids, pids]


train_df, valid_df = read_data_df()
train_mat = read_data_matrix(train_df)
train_mat = impute_values(train_mat)

### Singular value spectrum

singular_values = np.linalg.svd(train_mat, compute_uv=False, hermitian=False)
plt.plot(singular_values)
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Singular value spectrum")
# plt.show()

train_recon = opt_rank_k_approximation(train_mat, k=2)


pred_fn = lambda sids, pids: matrix_pred_fn(train_recon, sids, pids)

# Evaluate on validation data
val_score = evaluate(valid_df, pred_fn)
print(f"Validation RMSE: {val_score:.3f}")
make_submission(pred_fn, "svd_submission.csv")


# ## MY

# train_df, valid_df = read_data_df()
# train_mat = read_data_matrix(train_df)

# # train_mat = train_mat.T

# mask = ~np.isnan(train_mat)
# train_recon = np.zeros_like(train_mat)

# # exit()

# for i in trange(train_mat.shape[0]):
#     nsd = -(train_mat - train_mat[i]) ** 2
#     ll = (-16 + np.nansum(nsd, axis=1)) / (1 + (~np.isnan(nsd)).sum(axis=1))
#     w = np.exp(ll)
#     w[i] = 0
#     w = w[:, None] * mask
#     train_recon[i] = (w * np.nan_to_num(train_mat)).sum(axis=0) / w.sum(axis=0)

# # train_recon = train_recon.T

# pred_fn = lambda sids, pids: matrix_pred_fn(train_recon, sids, pids)

# # Evaluate on validation data
# val_score = evaluate(valid_df, pred_fn)
# print(f"Validation RMSE: {val_score:.3f}")
# make_submission(pred_fn, "svd_submission.csv")

# exit()

## Learned embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
class EmbeddingDotProductModel(nn.Module):
    def __init__(self, num_scientists: int, num_papers: int, dim: int):
        super().__init__()

        # Assign to each scientist and paper an embedding
        self.scientist_emb = nn.Embedding(num_scientists, dim)
        self.paper_emb = nn.Embedding(num_papers, dim)
        
    def forward(self, sid: torch.Tensor, pid: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            sid: [B,], int
            pid: [B,], int
        
        Outputs: [B,], float
        """

        # Per-pair dot product
        return torch.sum(self.scientist_emb(sid) * self.paper_emb(pid), dim=-1)

class RegressionNorm(nn.Module):
    def __init__(self, num_scientists: int, num_papers: int, dim: int):
        super().__init__()

        # Assign to each scientist and paper an embedding
        self.dim = dim
        self.scientist_emb = nn.Embedding(num_scientists, dim)
        self.paper_emb = nn.Embedding(num_papers, dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.LeakyReLU(1e-3),
            nn.Linear(dim, dim),
            nn.LeakyReLU(1e-3),
            nn.Linear(dim, 1),
        )
        
        p_mean, p_std = np.load('rating_mean_std_p.npy').astype(np.float32)
        self.p_mean = torch.from_numpy(p_mean).to(device)
        self.p_std = torch.from_numpy(p_std).to(device)
        
        s_mean, s_std = np.load('rating_mean_std_s.npy').astype(np.float32)
        self.s_mean = torch.from_numpy(s_mean).to(device)
        self.s_std = torch.from_numpy(s_std).to(device)
        
        # self.s_approx = torch.from_numpy(np.load('onerank-approx-s.npy')).float().to(device)
        # self.p_approx = torch.from_numpy(np.load('onerank-approx-p.npy')).float().to(device)

    def forward(self, sid: torch.Tensor, pid: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            sid: [B,], int
            pid: [B,], int
        
        Outputs: [B,], float
        """

        # Per-pair dot product
        s_emb = self.scientist_emb(sid)
        p_emb = self.paper_emb(pid)
        # y = torch.sum(s_emb * p_emb, dim=-1) / self.dim 
        x = torch.cat((s_emb, p_emb), dim=-1)
        x = self.mlp(x)
        # x = F.sigmoid(x) * 6
        x = x.squeeze()

        # x = self.p_mean[pid] + x * self.p_std[pid]
        x = self.s_mean[sid] + x * self.s_std[sid]
        # x = x + self.s_approx[sid] * self.p_approx[pid]
        
        # m = (self.s_mean[sid] * self.p_mean[pid]) ** 0.5
        # s = (self.s_std[sid] + self.p_std[pid]) / 2
        # x = x * s + m

        return x

# Define model (10k scientists, 1k papers, 32-dimensional embeddings) and optimizer
# model = EmbeddingDotProductModel(10_000, 1_000, 8).to(device)
model = RegressionNorm(10_000, 1_000, 32).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-2)
# optim = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)

def get_dataset(df: pd.DataFrame) -> torch.utils.data.Dataset:
    """Conversion from pandas data frame to torch dataset."""
    
    sids = torch.from_numpy(df["sid"].to_numpy())
    pids = torch.from_numpy(df["pid"].to_numpy())
    ratings = torch.from_numpy(df["rating"].to_numpy()).float()
    return torch.utils.data.TensorDataset(sids, pids, ratings)

train_dataset = get_dataset(train_df)
valid_dataset = get_dataset(valid_df)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2**8, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2**8, shuffle=False)

NUM_EPOCHS = 4
for epoch in range(NUM_EPOCHS):
    # Train model for an epoch
    total_loss = 0.0
    total_data = 0
    model.train()
    for sid, pid, ratings in tqdm(train_loader):
        # Move data to GPU
        sid = sid.to(device)
        pid = pid.to(device)
        ratings = ratings.to(device)

        # Make prediction and compute loss
        pred = model(sid, pid)
        loss = F.mse_loss(pred, ratings)

        # Compute gradients w.r.t. loss and take a step in that direction
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Keep track of running loss
        total_data += len(sid)
        total_loss += len(sid) * loss.item()

    # Evaluate model on validation data
    total_val_mse = 0.0
    total_val_data = 0
    model.eval()
    for sid, pid, ratings in valid_loader:
        # Move data to GPU
        sid = sid.to(device)
        pid = pid.to(device)
        ratings = ratings.to(device)

        # Clamp predictions in [1,5], since all ground-truth ratings are
        pred = model(sid, pid)
        pred = pred.clamp(1, 5)
        mse = F.mse_loss(pred, ratings)

        # Keep track of running metrics
        total_val_data += len(sid)
        total_val_mse += len(sid) * mse.item()

    print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Train loss={total_loss / total_data:.3f}, Valid RMSE={(total_val_mse / total_val_data) ** 0.5:.3f}")


pred_fn = lambda sids, pids: model(torch.from_numpy(sids).to(device), torch.from_numpy(pids).to(device)).clamp(1, 5).cpu().numpy()

# Evaluate on validation data
with torch.no_grad():
    val_score = evaluate(valid_df, pred_fn)

print(f"Validation RMSE: {val_score:.3f}")
with torch.no_grad():
    # make_submission(pred_fn, f"emb4_epoch{NUM_EPOCHS}.csv")
    make_submission(pred_fn, f"emb32_mlp32-32_snorm_epoch{NUM_EPOCHS}.csv")

## Outlook


