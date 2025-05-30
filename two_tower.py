import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.data_utils import read_data_df, read_data_matrix, impute_values, get_wishlist_matrix, get_wishlist_dict, evaluate, make_submission
import argparse

parser = argparse.ArgumentParser(description="two tower")
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--nce', type=int, default=1, help='use nce or not')

args = parser.parse_args()

SEED = args.seed
torch.manual_seed(SEED)
np.random.seed(SEED)

train_df, valid_df = read_data_df()
train_mat = read_data_matrix(train_df)
train_mat = impute_values(train_mat)
wishlist_mat = get_wishlist_matrix()
wishlist_dict = get_wishlist_dict()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

class RegressionNorm(nn.Module):
    def __init__(self, num_scientists: int, num_papers: int, initial_embed_dim: int, latent_dim: int, dropout_rate: float = 0.1):
        super().__init__()

        # Assign to each scientist and paper an embedding
        dim = initial_embed_dim
        self.scientist_emb = nn.Embedding(num_scientists, dim)
        self.paper_emb = nn.Embedding(num_papers, dim)

        self.wishlist_emb = nn.Embedding.from_pretrained(torch.from_numpy(get_wishlist_matrix().astype(np.float32)), freeze=True)
        self.wishlist_layer = nn.Linear(1_000, dim, bias=False)

        self.scientist_bias = nn.Embedding(num_scientists, 1)
        self.paper_bias = nn.Embedding(num_papers, 1)


        self.scientist_encoder = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(1e-3),
            nn.LayerNorm(dim),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LeakyReLU(1e-3),
            nn.LayerNorm(dim),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, latent_dim),
            # nn.LeakyReLU(1e-3)
        )
        self.paper_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(1e-3),
            nn.LayerNorm(dim),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LeakyReLU(1e-3),
            nn.LayerNorm(dim),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, latent_dim),
            # nn.LeakyReLU(1e-3)
        )

        self.mlp = nn.Sequential(
            nn.LayerNorm(latent_dim * 4),
            nn.Linear(latent_dim * 4, dim),
            nn.LeakyReLU(1e-3),
            nn.LayerNorm(dim),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LeakyReLU(1e-3),
            nn.LayerNorm(dim),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, 1),
            # nn.Softmax(dim=-1)
        )

        s_stats = np.load('rating_mean_std_s.npy').astype(np.float32)
        
        s_mean, s_std = s_stats[:, 0], s_stats[:, 1]
        self.s_mean = torch.from_numpy(s_mean).to(device)
        self.s_std = torch.from_numpy(s_std).to(device)

        


    def forward(self, sid: torch.Tensor, pid: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            sid: [B,], int
            pid: [B,], int
        
        Outputs: [B,], float
        """

        s_emb = self.scientist_emb(sid)
        p_emb = self.paper_emb(pid)

        # Per-pair dot product
        w_emb = self.wishlist_layer(self.wishlist_emb(sid))
        s_input = torch.cat([s_emb, w_emb], dim=-1)
        s_latent = self.scientist_encoder(s_input)
        p_latent = self.paper_encoder(p_emb)

        # s_latent = s_latent + torch.randn_like(s_latent) * self.epsilon
        # p_latent = p_latent + torch.randn_like(p_latent) * self.epsilon

        s_latent_norm = F.normalize(s_latent, dim=-1)
        p_latent_norm = F.normalize(p_latent, dim=-1)


        x = torch.cat([s_latent_norm, p_latent_norm, s_latent_norm * p_latent_norm, torch.abs(s_latent_norm - p_latent_norm)], dim=-1)

        x = self.mlp(x)
        x = x.squeeze()

        x += self.scientist_bias(sid).squeeze() + self.paper_bias(pid).squeeze()

        return x, s_emb, p_emb, s_latent_norm, p_latent_norm, s_latent_norm, p_latent_norm

# Compute mean and std for each scientist (sid)
sid_stats = train_df.groupby("sid")["rating"].agg(["mean", "std"]).reindex(range(10_000), fill_value=np.nan)
# Fill NaNs with global mean and std if any sid is missing
global_mean = train_df["rating"].mean()
global_std = train_df["rating"].std()
sid_stats["mean"].fillna(global_mean, inplace=True)
sid_stats["std"].fillna(global_std, inplace=True)
# Save to file
np.save('rating_mean_std_s.npy', sid_stats[["mean", "std"]].values.astype(np.float32))



def get_dataset(df: pd.DataFrame) -> torch.utils.data.Dataset:
    """Conversion from pandas data frame to torch dataset."""
    
    sids = torch.from_numpy(df["sid"].to_numpy())
    pids = torch.from_numpy(df["pid"].to_numpy())
    ratings = torch.from_numpy(df["rating"].to_numpy()).float()
    return torch.utils.data.TensorDataset(sids, pids, ratings)

train_dataset = get_dataset(train_df)
valid_dataset = get_dataset(valid_df)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2**12, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2**12, shuffle=False)

model = RegressionNorm(10_000, 1_000, 512, 150, dropout_rate=0.1).to(device)
nn.init.zeros_(model.scientist_bias.weight)
nn.init.zeros_(model.paper_bias.weight)

optim = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-4)

contrastive_weight = 0.1

NUM_EPOCHS = 30
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2)

def info_nce_loss(s_latent, p_latent, ratings, threshold=4.0, temperature=0.07):
    pos_mask = ratings >= threshold
    s_pos = s_latent[pos_mask]
    p_pos = p_latent[pos_mask]
    
    if len(s_pos) < 2:  # Not enough for contrastive batch
        return torch.tensor(0.0, device=s_latent.device)

    logits = torch.matmul(s_pos, p_pos.T) / temperature
    labels = torch.arange(len(s_pos), device=logits.device)
    return F.cross_entropy(logits, labels)

def multithreshold_infonce_loss(latents1, latents2, ratings, thresholds=[4, 5], temperatures=[0.07, 0.07]):
    losses = []
    for th, temp in zip(thresholds, temperatures):
        # print(losses)
        losses.append(info_nce_loss(latents1, latents2, ratings, threshold=th, temperature=temp))
    return torch.stack(losses).mean()

best_val_rmse = float('inf')
best_epoch = -1
best_model_state = None

val_rmse_per_epoch = []
for epoch in range(NUM_EPOCHS):
    # Train model for an epoch
    total_loss = 0.0
    total_data = 0
    avg_rating_loss = 0
    avg_contrastive_loss = 0
    model.train()
    for sid, pid, ratings in tqdm(train_loader):
        # Move data to GPU
        sid = sid.to(device)
        pid = pid.to(device)
        ratings = ratings.to(device)

        # Make prediction and compute loss
        pred, s_emb, p_emb, s_out, p_out, s_latent, p_latent = model(sid, pid)
        rating_loss = F.mse_loss(pred.clamp(1, 5), ratings)
        

        # Contrastive loss for latent vectors based on rating
        if(args.nce == 1):
            contrastive_loss = info_nce_loss(s_latent, p_latent, ratings) + info_nce_loss(p_latent, s_latent, ratings)
        elif(args.nce == 2):
            contrastive_loss = multithreshold_infonce_loss(s_latent, p_latent, ratings) + multithreshold_infonce_loss(p_latent, s_latent, ratings)
        else:
            contrastive_loss = 0
        # contrastive_loss = torch.tensor(0.0, device=sid.device)
        
        loss = rating_loss +  contrastive_weight * contrastive_loss

        # Compute gradients w.r.t. loss and take a step in that direction
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Keep track of running loss
        total_data += len(sid)
        total_loss += len(sid) * loss.item()
        avg_rating_loss += rating_loss.item() / len(train_dataset) * len(sid)
        avg_contrastive_loss += contrastive_loss.item() / len(train_dataset) * len(sid)

    scheduler.step()
    print(f"[Epoch {epoch+1}] rating={avg_rating_loss:.4f}, contrastive={avg_contrastive_loss:.4f}")

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
        pred = pred[0].clamp(1, 5)
        
        mse = F.mse_loss(pred, ratings)

        # Keep track of running metrics
        total_val_data += len(sid)
        total_val_mse += len(sid) * mse.item()

    val_rmse = (total_val_mse / total_val_data) ** 0.5
    val_rmse_per_epoch.append(val_rmse)
    print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Train loss={total_loss / total_data:.3f}, Valid RMSE={val_rmse:.3f}")

    # Save best model
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_epoch = epoch + 1
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

# Restore best model state
if best_model_state is not None:
    model.load_state_dict(best_model_state)

def pred_fn(sids, pids, batch_size=4096):
    preds = []
    sids = torch.from_numpy(sids)
    pids = torch.from_numpy(pids)
    for i in range(0, len(sids), batch_size):
        sid_batch = sids[i:i+batch_size].to(device)
        pid_batch = pids[i:i+batch_size].to(device)
        with torch.no_grad():
            pred_batch = model(sid_batch, pid_batch)[0].clamp(1, 5).cpu().numpy()
        preds.append(pred_batch)
    return np.concatenate(preds)

# Evaluate on validation data
with torch.no_grad():
    val_score = evaluate(valid_df, pred_fn)

print(f"Validation RMSE: {val_score:.3f}")
with torch.no_grad():
    make_submission(pred_fn, f"contrast_epochs{NUM_EPOCHS}_seed{SEED}_nce_type{args.nce}.csv")



