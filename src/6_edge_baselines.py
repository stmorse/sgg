import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score

from models import SimpleGCN
from utils import parse_windows

BASE_PATH = '/sciclone/geograd/stmorse/reddit/subreddit/science/filtered'

def sample_negative_edges(n_nodes, pos_set, n_samples, device):
    neg = set()
    while len(neg) < n_samples:
        cand = torch.randint(0, n_nodes, (n_samples*2,2), device=device).tolist()
        for u, v in cand:
            if u == v or (u,v) in pos_set or (v,u) in pos_set:
                continue
            neg.add((u,v))
            if len(neg) >= n_samples:
                break
    src = torch.tensor([u for u,_ in neg], dtype=torch.long, device=device)
    dst = torch.tensor([v for _,v in neg], dtype=torch.long, device=device)
    return torch.stack([src, dst], dim=0)

def safe_auc(y_true, y_score):
    yt = y_true.cpu().numpy() if torch.is_tensor(y_true) else np.array(y_true)
    ys = y_score.cpu().numpy() if torch.is_tensor(y_score) else np.array(y_score)
    if np.unique(yt).size < 2:
        return float('nan')
    return roc_auc_score(yt, ys)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 1) build window‐pairs
    wins  = parse_windows(
        args.start_year, args.start_month,
        args.end_year,   args.end_month,
        args.period
    )
    pairs = list(zip(wins, wins[1:]))
    if len(pairs) < args.num_test + 2:
        raise ValueError("Not enough window‐pairs for that many test splits")

    # train vs test split
    train_pairs = pairs[:-args.num_test]
    test_pairs  = pairs[-args.num_test:]

    # infer dims from first train window
    w0 = train_pairs[0][0]
    fx0 = f"features_{w0[0]}-{w0[1]:02d}_{w0[2]}-{w0[3]:02d}_filtered.npy"
    x0 = np.load(os.path.join(BASE_PATH, fx0))
    num_nodes, in_dim = x0.shape

    # preload all train data
    train_data = []
    for (y0,m0,y1,m1), (yn0,mn0,yn1,mn1) in train_pairs:
        tag0 = f"{y0}-{m0:02d}_{y1}-{m1:02d}"
        tag1 = f"{yn0}-{mn0:02d}_{yn1}-{mn1:02d}"

        # t0 graph
        g0 = json.load(open(os.path.join(BASE_PATH, f"graph_{tag0}_filtered.json")))
        ei0 = torch.tensor(g0['edge_index'], dtype=torch.long, device=device)
        pos0 = set(zip(ei0[0].tolist(), ei0[1].tolist()))
        n_pos = ei0.size(1)

        # t1 graph for naive baseline
        g1 = json.load(open(os.path.join(BASE_PATH, f"graph_{tag1}_filtered.json")))
        pos1 = set(zip(*g1['edge_index']))

        # features at t0
        x_t = torch.tensor(
            np.load(os.path.join(BASE_PATH, f"features_{tag0}_filtered.npy")),
            dtype=torch.float, device=device
        )

        # sample negatives at t0
        neg0 = sample_negative_edges(
            num_nodes, pos0,
            int(n_pos * args.neg_ratio),
            device
        )

        # candidate edges & labels
        edge_all = torch.cat([ei0, neg0], dim=1)
        labels  = torch.cat([
            torch.ones(n_pos, device=device),
            torch.zeros(neg0.size(1), device=device)
        ], dim=0)

        train_data.append({
            'tag0': tag0, 'tag1': tag1,
            'ei0': ei0, 'x0': x_t,
            'us': edge_all[0], 'vs': edge_all[1],
            'labels': labels, 'pos1': pos1
        })

    # 2) expanding‐window CV on train_pairs
    gcn_final = None
    for k in range(1, len(train_data)):
        subset = train_data[:k]
        val    = train_data[k]
        print(f"\n=== Fold {k}: train on windows [0…{k-1}], validate on {val['tag0']} → {val['tag1']} ===")

        # fresh GCN
        gcn = SimpleGCN(in_dim, args.hidden_dim).to(device)
        opt = torch.optim.Adam(gcn.parameters(), lr=args.lr)

        best_auc = 0.0
        for epoch in range(1, args.epochs+1):
            gcn.train()
            opt.zero_grad()
            for d in subset:
                h = gcn(d['x0'], d['ei0'])
                scores = (h[d['us']] * h[d['vs']]).sum(1).sigmoid()
                loss = F.binary_cross_entropy(scores, d['labels'])
                loss.backward()
            opt.step()

            # evaluate on val
            gcn.eval()
            with torch.no_grad():
                h_val = gcn(val['x0'], val['ei0'])
                scores_val = (h_val[val['us']] * h_val[val['vs']]).sum(1).sigmoid()
                auc_val = safe_auc(val['labels'], scores_val)
            best_auc = max(best_auc, auc_val)

        # naive persistence on val
        preds_naive = torch.tensor([
            1 if (u,v) in val['pos1'] or (v,u) in val['pos1'] else 0
            for u,v in zip(val['us'].tolist(), val['vs'].tolist())
        ], dtype=torch.float, device=device)
        auc_naive = safe_auc(val['labels'], preds_naive)

        print(f"  Naive AUC={auc_naive:.4f} | GCN best‐val AUC={best_auc:.4f}")

        # keep the last model (trained on full training history)
        if k == len(train_data) - 1:
            gcn_final = gcn

    # 3) final test on held‐out windows
    print("\n=== TEST SET ===")
    gcn_final.eval()
    for (y0,m0,y1,m1), (yn0,mn0,yn1,mn1) in test_pairs:
        tag0 = f"{y0}-{m0:02d}_{y1}-{m1:02d}"
        tag1 = f"{yn0}-{mn0:02d}_{yn1}-{mn1:02d}"
        # load t0 graph+features
        g0 = json.load(open(os.path.join(BASE_PATH, f"graph_{tag0}_filtered.json")))
        ei0 = torch.tensor(g0['edge_index'], dtype=torch.long, device=device)
        pos0 = set(zip(ei0[0].tolist(), ei0[1].tolist()))
        n_pos = ei0.size(1)
        x0 = torch.tensor(
            np.load(os.path.join(BASE_PATH, f"features_{tag0}_filtered.npy")),
            dtype=torch.float, device=device
        )
        # load t1 for naive
        g1 = json.load(open(os.path.join(BASE_PATH, f"graph_{tag1}_filtered.json")))
        pos1 = set(zip(*g1['edge_index']))
        # sample negatives t0
        neg0 = sample_negative_edges(
            num_nodes, pos0,
            int(n_pos * args.neg_ratio),
            device
        )
        edge_all = torch.cat([ei0, neg0], dim=1)
        labels   = torch.cat([
            torch.ones(n_pos, device=device),
            torch.zeros(neg0.size(1), device=device)
        ], dim=0)

        # GCN predictions
        with torch.no_grad():
            h = gcn_final(x0, ei0)
            scores = (h[edge_all[0]] * h[edge_all[1]]).sum(1).sigmoid()
            auc_gcn = safe_auc(labels, scores)

        # naive persistence
        preds_naive = torch.tensor([
            1 if (u,v) in pos1 or (v,u) in pos1 else 0
            for u,v in zip(edge_all[0].tolist(), edge_all[1].tolist())
        ], dtype=torch.float, device=device)
        auc_naive = safe_auc(labels, preds_naive)

        print(f"[TEST {tag0}→{tag1}] Naive AUC={auc_naive:.4f} | GCN AUC={auc_gcn:.4f}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--start_year',   type=int,   required=True)
    p.add_argument('--start_month',  type=int,   default=1)
    p.add_argument('--end_year',     type=int,   required=True)
    p.add_argument('--end_month',    type=int,   default=12)
    p.add_argument('--period',       type=int,   required=True,
                   help='window length in months')
    p.add_argument('--hidden_dim',   type=int,   default=32)
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--lr',           type=float, default=0.01)
    p.add_argument('--neg_ratio',    type=float, default=1.0)
    p.add_argument('--num_test',     type=int,   default=2,
                   help='number of final windows to hold out for test')
    args = p.parse_args()

    main(args)