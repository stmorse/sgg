#!/usr/bin/env python3
"""
Baseline feature-prediction over arbitrary-length month windows.

For each consecutive window-pair (t → t+1), it runs three baselines:
1) Naive persistence: X_{t+1} = X_t
2) Linear regression: fit W to map X_t -> X_{t+1}
3) GCN: 2-layer GCN to predict features

Reports MSE on a train/validation node split per window.
Usage:
  python baseline_feature_pred_range.py \
    --start_year 2007 --start_month 1 \
    --end_year 2008   --end_month 12 \
    --period 6 \
    --train_ratio 0.85 --val_ratio 0.15 \
    --hidden_dim 32 --epochs 50 --lr 0.01
"""
import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Base path for filtered graphs & features
BASE_PATH = '/sciclone/geograd/stmorse/reddit/subreddit/science/filtered'


def month_index(year, month):
    return year * 12 + (month - 1)

def idx_to_ym(idx):
    return idx // 12, (idx % 12) + 1


def parse_windows(sy, sm, ey, em, period):
    start = month_index(sy, sm)
    end   = month_index(ey, em)
    wins = []
    i = start
    while i <= end:
        j = min(i + period - 1, end)
        y0, m0 = idx_to_ym(i)
        y1, m1 = idx_to_ym(j)
        wins.append((y0, m0, y1, m1))
        i += period
    return wins


def split_node_masks(n_nodes, train_ratio, val_ratio, seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(n_nodes)
    t_end = int(train_ratio * n_nodes)
    v_end = t_end + int(val_ratio * n_nodes)
    train_idx = idx[:t_end]
    val_idx   = idx[t_end:v_end]
    train_mask = np.zeros(n_nodes, bool)
    val_mask   = np.zeros(n_nodes, bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    return train_mask, val_mask


class GCNFeat(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)


def run_baselines(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    windows = parse_windows(
        args.start_year, args.start_month,
        args.end_year,   args.end_month,
        args.period
    )
    pairs = list(zip(windows, windows[1:]))
    print("Window-pairs:", pairs)

    naive_mses, lr_mses, gcn_mses = [], [], []

    for (y0, m0, y1, m1), (yn0, mn0, yn1, mn1) in pairs:
        tag0 = f"{y0}-{m0:02d}_{y1}-{m1:02d}"
        tag1 = f"{yn0}-{mn0:02d}_{yn1}-{mn1:02d}"

        # Load adjacency for GCN
        g0 = json.load(open(os.path.join(BASE_PATH, f"graph_{tag0}_filtered.json")))
        edge_index = torch.tensor(g0['edge_index'], dtype=torch.long, device=device)

        # Load feature matrices
        f0 = np.load(os.path.join(BASE_PATH, f"features_{tag0}_filtered.npy"))
        f1 = np.load(os.path.join(BASE_PATH, f"features_{tag1}_filtered.npy"))
        num_nodes, dim = f0.shape

        # Split nodes
        train_mask, val_mask = split_node_masks(num_nodes, args.train_ratio, args.val_ratio)
        tr_idx = np.where(train_mask)[0]
        va_idx = np.where(val_mask)[0]

        # --- 1) Naive persistence ---
        mse_naive_train = mean_squared_error(f1[tr_idx], f0[tr_idx])
        mse_naive_val   = mean_squared_error(f1[va_idx], f0[va_idx])

        # --- 2) Linear regression ---
        lr = LinearRegression()
        lr.fit(f0[tr_idx], f1[tr_idx])
        pred_lr_tr = lr.predict(f0[tr_idx])
        pred_lr_va = lr.predict(f0[va_idx])
        mse_lr_tr = mean_squared_error(f1[tr_idx], pred_lr_tr)
        mse_lr_va = mean_squared_error(f1[va_idx], pred_lr_va)

        # --- 3) GCN ---
        x0 = torch.tensor(f0, dtype=torch.float, device=device)
        x1 = torch.tensor(f1, dtype=torch.float, device=device)
        model = GCNFeat(in_dim=dim, hidden_dim=args.hidden_dim, out_dim=dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        crit = torch.nn.MSELoss()

        best_val = float('inf')
        for ep in range(1, args.epochs+1):
            model.train(); opt.zero_grad()
            out = model(x0, edge_index)
            loss = crit(out[tr_idx], x1[tr_idx])
            loss.backward(); opt.step()
            # val
            model.eval()
            with torch.no_grad():
                val_loss = crit(out[va_idx], x1[va_idx]).item()
            if val_loss < best_val:
                best_val = val_loss
        mse_gcn = best_val

        naive_mses.append(mse_naive_val)
        lr_mses.append(mse_lr_va)
        gcn_mses.append(mse_gcn)

        print(
            f"Period {tag0}->{tag1}: "
            f"Naive MSE (tr/va)={mse_naive_train:.4f}/{mse_naive_val:.4f}  "
            f"Linear MSE={mse_lr_va:.4f}  "
            f"GCN MSE={mse_gcn:.4f}"
        )
    
    print(f'Mean MSE Naive: {np.mean(naive_mses)} | Linear: {np.mean(lr_mses)} | GCN {np.mean(gcn_mses)}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start_year',   type=int, required=True)
    p.add_argument('--start_month',  type=int, default=1)
    p.add_argument('--end_year',     type=int, required=True)
    p.add_argument('--end_month',    type=int, default=12)
    p.add_argument('--period',       type=int, required=True)
    p.add_argument('--train_ratio',  type=float, default=0.85)
    p.add_argument('--val_ratio',    type=float, default=0.15)
    p.add_argument('--hidden_dim',   type=int, default=32)
    p.add_argument('--epochs',       type=int, default=50)
    p.add_argument('--lr',           type=float, default=0.01)
    args = p.parse_args()
    run_baselines(args)

if __name__=='__main__':
    main()
