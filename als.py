import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

import numpy as np
from sklearn.metrics import root_mean_squared_error
from utils.data_utils import read_data_df, read_data_matrix, get_wishlist_matrix, get_wishlist_dict, impute_values

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


train_df, valid_df = read_data_df()
n_users = 10_000
n_items = 1_000
latent_dim = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Torch datasets
def df_to_dataset(df):
    return TensorDataset(
        torch.tensor(df['sid'].values, dtype=torch.long),
        torch.tensor(df['pid'].values, dtype=torch.long),
        torch.tensor(df['rating'].values, dtype=torch.float32)
    )

train_loader = DataLoader(df_to_dataset(train_df), batch_size=4096, shuffle=True)
val_loader = DataLoader(df_to_dataset(valid_df), batch_size=8192)

class ALSModel(nn.Module):
    def __init__(self, n_users, n_items, k=64):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, k)
        self.item_factors = nn.Embedding(n_items, k)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        # nn.init.xavier_uniform_(self.user_factors.weight)
        # nn.init.xavier_uniform_(self.item_factors.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users, items):
        u = self.user_factors(users)
        v = self.item_factors(items)
        dot = (u * v).sum(dim=1)
        return dot + self.user_bias(users).squeeze() + self.item_bias(items).squeeze()


model = ALSModel(n_users, n_items, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10, gamma=0.1)

loss_fn = nn.MSELoss()

# Training loop
for epoch in range(30):
    model.train()
    total_loss = 0
    for sid, pid, rating in tqdm(train_loader):
        optimizer.zero_grad()
        sid, pid, rating = sid.to(device), pid.to(device), rating.to(device)
        # print(rating.shape, sid.shape, pid.shape)
        pred = model(sid, pid)
        # print(pred)
        loss = loss_fn(pred, rating)
        
        loss.backward()
        # print(loss.item())
        optimizer.step()
        total_loss += loss.item() * len(sid)

    scheduler.step()
    
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for sid, pid, rating in val_loader:
            sid, pid = sid.to(device), pid.to(device)
            pred = model(sid, pid).clamp(1, 5).cpu().numpy()
            preds.append(pred)
            truths.append(rating.numpy())
    val_rmse = root_mean_squared_error(np.concatenate(truths), np.concatenate(preds))
    print(f"[Epoch {epoch+1}] Train Loss={total_loss / len(train_df):.4f}, Val RMSE={val_rmse:.4f}")