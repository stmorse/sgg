#!/usr/bin/env python3
"""
Baseline feature‐prediction over arbitrary‐length month windows.

For each consecutive window‐pair (t → t+1), we:
 1) Hold out the last two period‐pairs as a pure test set.
 2) For k = 1 … K_train:
    • Fully retrain three baselines on windows 0…k−1:
        a) Naive persistence
        b) LinearRegression (cumulative)
        c) 2‐layer GCN
    • Print train/val MSE for each window in 0…k−1.
 3) Finally, refit LR & GCN on all K_train windows and evaluate on the two held‐out test windows.
Usage:
  python baseline_feature_pred_range.py \
    --start_year 2007 --start_month 1 \
    --end_year   2008 --end_month   12 \
    --period 6 \
    --train_ratio 0.85 --val_ratio 0.15 \
    --hidden_dim 32 --epochs 50 --lr 0.01
"""
import os, json, argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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
        wins.append((*idx_to_ym(i), *idx_to_ym(j)))
        i += period
    return wins

def split_node_masks(n, train_ratio, val_ratio, seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(n)
    t_end = int(train_ratio * n)
    v_end = t_end + int(val_ratio * n)
    tr_idx = idx[:t_end]
    va_idx = idx[t_end:v_end]
    tr_mask = np.zeros(n, bool); tr_mask[tr_idx] = True
    va_mask = np.zeros(n, bool); va_mask[va_idx] = True
    return tr_mask, va_mask

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
    # build period windows and pairs
    wins  = parse_windows(
        args.start_year, args.start_month,
        args.end_year,   args.end_month,
        args.period
    )
    pairs = list(zip(wins, wins[1:]))
    if len(pairs) < 3:
        raise ValueError("Need at least 3 period-pairs to hold out 2 for test.")

    # split into train/val history vs test
    num_test    = 2
    train_pairs = pairs[:-num_test]
    test_pairs  = pairs[-num_test:]

    # 1) load all train/val windows
    train_data = []
    for (y0,m0,y1,m1), (yn0,mn0,yn1,mn1) in train_pairs:
        tag0 = f"{y0}-{m0:02d}_{y1}-{m1:02d}"
        tag1 = f"{yn0}-{mn0:02d}_{yn1}-{mn1:02d}"

        # load graph & features
        g0 = json.load(open(os.path.join(BASE_PATH, f"graph_{tag0}_filtered.json")))
        edge_index = torch.tensor(g0['edge_index'], dtype=torch.long, device=device)
        f0 = np.load(os.path.join(BASE_PATH, f"features_{tag0}_filtered.npy"))
        f1 = np.load(os.path.join(BASE_PATH, f"features_{tag1}_filtered.npy"))
        N, dim = f0.shape

        # node split
        tr_mask, va_mask = split_node_masks(N, args.train_ratio, args.val_ratio)
        tr_idx = np.where(tr_mask)[0]
        va_idx = np.where(va_mask)[0]

        train_data.append({
            'tag0': tag0, 'tag1': tag1,
            'f0': f0, 'f1': f1,
            'tr_idx': tr_idx, 'va_idx': va_idx,
            'edge_index': edge_index,
            'x0': torch.tensor(f0, dtype=torch.float, device=device),
            'x1': torch.tensor(f1, dtype=torch.float, device=device),
        })

    # dimension for GCN
    dim = train_data[0]['f0'].shape[1]

    # 2) expanding-window full retrain
    for k in range(1, len(train_data)+1):
        subset = train_data[:k]
        last_tag1 = subset[-1]['tag1']
        print(f"\n=== Trained on t0→{last_tag1} ({k} windows) ===")

        # 2a) Naive persistence
        for d in subset:
            m_tr = mean_squared_error(d['f1'][d['tr_idx']], d['f0'][d['tr_idx']])
            m_va = mean_squared_error(d['f1'][d['va_idx']], d['f0'][d['va_idx']])
            print(f"[{d['tag0']}→{d['tag1']}] Naive Train={m_tr:.4f}, Val={m_va:.4f}")

        # 2b) Linear Regression
        Xc = np.vstack([d['f0'][d['tr_idx']] for d in subset])
        yc = np.vstack([d['f1'][d['tr_idx']] for d in subset])
        lr = LinearRegression().fit(Xc, yc)
        for d in subset:
            p_tr = lr.predict(d['f0'][d['tr_idx']])
            p_va = lr.predict(d['f0'][d['va_idx']])
            m_tr = mean_squared_error(d['f1'][d['tr_idx']], p_tr)
            m_va = mean_squared_error(d['f1'][d['va_idx']], p_va)
            print(f"[{d['tag0']}→{d['tag1']}]  LR Train={m_tr:.4f}, Val={m_va:.4f}")

        # 2c) GCN
        model = GCNFeat(in_dim=dim, hidden_dim=args.hidden_dim, out_dim=dim).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
        crit  = torch.nn.MSELoss()
        # train epochs
        for ep in range(1, args.epochs+1):
            model.train()
            for d in subset:
                opt.zero_grad()
                out = model(d['x0'], d['edge_index'])
                loss = crit(out[d['tr_idx']], d['x1'][d['tr_idx']])
                loss.backward()
                opt.step()
        # eval
        model.eval()
        with torch.no_grad():
            for d in subset:
                out = model(d['x0'], d['edge_index'])
                m_tr = crit(out[d['tr_idx']], d['x1'][d['tr_idx']]).item()
                m_va = crit(out[d['va_idx']], d['x1'][d['va_idx']]).item()
                print(f"[{d['tag0']}→{d['tag1']}] GCN Train={m_tr:.4f}, Val={m_va:.4f}")

    # 3) Final test on held-out windows using last-trained models
    # Refit final LR
    X_full = np.vstack([d['f0'][d['tr_idx']] for d in train_data])
    y_full = np.vstack([d['f1'][d['tr_idx']] for d in train_data])
    lr_final = LinearRegression().fit(X_full, y_full)

    # Retrain final GCN
    model_final = GCNFeat(in_dim=dim, hidden_dim=args.hidden_dim, out_dim=dim).to(device)
    opt_f = torch.optim.Adam(model_final.parameters(), lr=args.lr)
    crit  = torch.nn.MSELoss()
    for ep in range(1, args.epochs+1):
        model_final.train()
        for d in train_data:
            opt_f.zero_grad()
            out = model_final(d['x0'], d['edge_index'])
            loss = crit(out[d['tr_idx']], d['x1'][d['tr_idx']])
            loss.backward()
            opt_f.step()

    model_final.eval()
    print("\n=== TEST SET ===")
    for (y0,m0,y1,m1), (yn0,mn0,yn1,mn1) in test_pairs:
        tag0 = f"{y0}-{m0:02d}_{y1}-{m1:02d}"
        tag1 = f"{yn0}-{mn0:02d}_{yn1}-{mn1:02d}"

        # load
        g0 = json.load(open(os.path.join(BASE_PATH, f"graph_{tag0}_filtered.json")))
        ei = torch.tensor(g0['edge_index'], dtype=torch.long, device=device)
        f0 = np.load(os.path.join(BASE_PATH, f"features_{tag0}_filtered.npy"))
        f1 = np.load(os.path.join(BASE_PATH, f"features_{tag1}_filtered.npy"))
        N = f0.shape[0]
        _, va_mask = split_node_masks(N, args.train_ratio, args.val_ratio)
        va_idx = np.where(va_mask)[0]

        # Naive
        m_naive = mean_squared_error(f1[va_idx], f0[va_idx])
        # LR
        m_lr    = mean_squared_error(f1[va_idx], lr_final.predict(f0[va_idx]))
        # GCN
        x0 = torch.tensor(f0, dtype=torch.float, device=device)
        x1 = torch.tensor(f1, dtype=torch.float, device=device)
        with torch.no_grad():
            out = model_final(x0, ei)
        m_gcn = crit(out[torch.tensor(va_idx, device=device)],
                     x1[torch.tensor(va_idx, device=device)]).item()

        print(f"[{tag0}→{tag1}]  Naive={m_naive:.4f}, LR={m_lr:.4f}, GCN={m_gcn:.4f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start_year',  type=int,   required=True)
    p.add_argument('--start_month', type=int,   default=1)
    p.add_argument('--end_year',    type=int,   required=True)
    p.add_argument('--end_month',   type=int,   default=12)
    p.add_argument('--period',      type=int,   required=True)
    p.add_argument('--train_ratio', type=float, default=0.85)
    p.add_argument('--val_ratio',   type=float, default=0.15)
    p.add_argument('--hidden_dim',  type=int,   default=32)
    p.add_argument('--epochs',      type=int,   default=50)
    p.add_argument('--lr',          type=float, default=0.01)
    args = p.parse_args()
    run_baselines(args)

if __name__ == '__main__':
    main()
