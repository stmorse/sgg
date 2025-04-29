#!/usr/bin/env python3
"""
Baseline graph-prediction over a date range of arbitrary-length windows.

Two baselines per window-pair:
1) Naive persistence: predict A_{t+1}(u,v)=1 iff A_t(u,v)=1
2) Simple GCN: one-layer GCN+dot decoder trained per window.

Usage:
  python baseline_graph_pred_range.py \
    --start_year 2007 --start_month 1 \
    --end_year 2008   --end_month 12 \
    --period 6 \
    --neg_ratio 1.0 --train_ratio 0.7 --val_ratio 0.15 \
    --hidden_dim 32 --epochs 50 --lr 0.01
"""
import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score

# Base path for filtered graphs & features\
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
        y0,m0 = idx_to_ym(i)
        y1,m1 = idx_to_ym(j)
        wins.append((y0,m0,y1,m1))
        i += period
    return wins


def sample_negative_edges(n_nodes, pos_set, n_samples, device):
    neg = set()
    while len(neg) < n_samples:
        cand = torch.randint(0, n_nodes, (n_samples*2,2), device=device).tolist()
        for u,v in cand:
            if u == v or (u,v) in pos_set or (v,u) in pos_set:
                continue
            neg.add((u,v))
            if len(neg) >= n_samples:
                break
    src = torch.tensor([u for u,v in neg], dtype=torch.long, device=device)
    dst = torch.tensor([v for u,v in neg], dtype=torch.long, device=device)
    return torch.stack([src,dst], dim=0)


def split_masks(n, train_ratio, val_ratio, seed=42, device=None):
    torch.manual_seed(seed)
    perm = torch.randperm(n, device=device)
    t_end = int(train_ratio * n)
    v_end = t_end + int(val_ratio * n)
    train = torch.zeros(n, dtype=torch.bool, device=device)
    val   = train.clone()
    test  = train.clone()
    train[perm[:t_end]] = True
    val[perm[t_end:v_end]] = True
    test[perm[v_end:]]   = True
    return train, val, test


class SimpleGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv = GCNConv(in_dim, hidden_dim)
    def forward(self, x, edge_index):
        h = F.relu(self.conv(x, edge_index))
        return h


def run_baselines(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    windows = parse_windows(
        args.start_year, args.start_month,
        args.end_year,   args.end_month,
        args.period
    )
    pairs = list(zip(windows, windows[1:]))
    print("Window-pairs:", pairs)

    for (y0,m0,y1,m1), (yn0,mn0,yn1,mn1) in pairs:
        tag0 = f"{y0}-{m0:02d}_{y1}-{m1:02d}"
        tag1 = f"{yn0}-{mn0:02d}_{yn1}-{mn1:02d}"
        # load graphs
        g0 = json.load(open(os.path.join(BASE_PATH, f"graph_{tag0}_filtered.json")))
        g1 = json.load(open(os.path.join(BASE_PATH, f"graph_{tag1}_filtered.json")))
        ei0 = torch.tensor(g0['edge_index'], dtype=torch.long, device=device)
        pos0 = set(zip(ei0[0].tolist(), ei0[1].tolist()))
        n_pos = ei0.size(1)
        pos1 = set(zip(*g1['edge_index']))
        # load features
        x0 = torch.tensor(
            np.load(os.path.join(BASE_PATH, f"features_{tag0}_filtered.npy")),
            dtype=torch.float, device=device
        )
        num_nodes, in_dim = x0.size()
        # build candidate edges
        neg0 = sample_negative_edges(num_nodes, pos0,
                                     int(n_pos*args.neg_ratio), device)
        edge_all = torch.cat([ei0, neg0], dim=1)
        labels = torch.cat([
            torch.ones(n_pos, device=device),
            torch.zeros(neg0.size(1), device=device)
        ])
        # split
        perm = torch.randperm(edge_all.size(1), device=device)
        edge_all = edge_all[:,perm]
        labels   = labels[perm]
        tr_m,va_m,_ = split_masks(edge_all.size(1),
                                  args.train_ratio, args.val_ratio,
                                  device=device)

        # --- Naive persistence ---
        preds_naive = torch.zeros_like(labels)
        us,vs = edge_all
        for i,(u,v) in enumerate(zip(us.tolist(), vs.tolist())):
            preds_naive[i] = 1 if (u,v) in pos1 or (v,u) in pos1 else 0
        auc_naive = roc_auc_score(labels[va_m].cpu(), preds_naive[va_m].cpu())

        # --- Simple GCN ---
        gcn = SimpleGCN(in_dim, args.hidden_dim).to(device)
        opt = torch.optim.Adam(gcn.parameters(), lr=args.lr)
        best_auc=0
        for epoch in range(1, args.epochs+1):
            gcn.train(); opt.zero_grad()
            h = gcn(x0, ei0)
            # decode
            us_tr = edge_all[:,tr_m]
            scores = (h[us_tr[0]] * h[us_tr[1]]).sum(1).sigmoid()
            loss = F.binary_cross_entropy(scores, labels[tr_m])
            loss.backward(); opt.step()
            # val
            gcn.eval()
            with torch.no_grad():
                us_va = edge_all[:,va_m]
                s_va = (h[us_va[0]] * h[us_va[1]]).sum(1).sigmoid()
                auc = roc_auc_score(labels[va_m].cpu(), s_va.cpu())
            best_auc = max(best_auc, auc)
        print(f"Period {tag0}->{tag1}  |  Naive AUC={auc_naive:.4f}  GCN AUC={best_auc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_year',   type=int, required=True)
    parser.add_argument('--start_month',  type=int, default=1)
    parser.add_argument('--end_year',     type=int, required=True)
    parser.add_argument('--end_month',    type=int, default=12)
    parser.add_argument('--period',       type=int, required=True)
    parser.add_argument('--neg_ratio',    type=float, default=1.0)
    parser.add_argument('--train_ratio',  type=float, default=0.7)
    parser.add_argument('--val_ratio',    type=float, default=0.15)
    parser.add_argument('--hidden_dim',   type=int, default=32)
    parser.add_argument('--epochs',       type=int, default=50)
    parser.add_argument('--lr',           type=float, default=0.01)
    args = parser.parse_args()
    run_baselines(args)

if __name__=='__main__':
    main()
