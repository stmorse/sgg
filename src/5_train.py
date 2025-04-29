#!/usr/bin/env python3
"""
Train edge-level GAE over arbitrary-month windows with negative sampling & combined loss.
"""
import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from models import EdgeEncoder, EdgeDecoder, FeatureDecoder

# Base directory for filtered data\
BASEPATH = '/sciclone/geograd/stmorse/reddit/subreddit/science/filtered'

def month_index(year, month):
    return year * 12 + (month - 1)

def idx_to_ym(idx):
    return idx // 12, (idx % 12) + 1


def sample_negative_edges(num_nodes, pos_set, num_samples, device):
    neg = set()
    while len(neg) < num_samples:
        # sample in batches
        cand = torch.randint(0, num_nodes, (num_samples * 2, 2), device=device)
        for u, v in cand.tolist():
            if u == v:
                continue
            if (u, v) in pos_set or (v, u) in pos_set:
                continue
            neg.add((u, v))
            if len(neg) >= num_samples:
                break
    src = torch.tensor([u for u, v in neg], dtype=torch.long, device=device)
    dst = torch.tensor([v for u, v in neg], dtype=torch.long, device=device)
    return torch.stack([src, dst], dim=0)


def split_masks(n, train_ratio, val_ratio, seed=42, device=None):
    torch.manual_seed(seed)
    idx = torch.randperm(n, device=device)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)
    train_mask = torch.zeros(n, dtype=torch.bool, device=device)
    val_mask = torch.zeros(n, dtype=torch.bool, device=device)
    test_mask = torch.zeros(n, dtype=torch.bool, device=device)
    train_mask[idx[:train_end]] = True
    val_mask[idx[train_end:val_end]] = True
    test_mask[idx[val_end:]] = True
    return train_mask, val_mask, test_mask


def parse_windows(start_year, start_month, end_year, end_month, period):
    idx0 = month_index(start_year, start_month)
    idx_end = month_index(end_year, end_month)
    windows = []
    i = idx0
    while i <= idx_end:
        j = min(i + period - 1, idx_end)
        y0, m0 = idx_to_ym(i)
        y1, m1 = idx_to_ym(j)
        windows.append((y0, m0, y1, m1))
        i += period
    return windows


def main():
    parser = argparse.ArgumentParser(
        description="Train edge-level GAE on arbitrary-length month windows"
    )
    parser.add_argument('--start_year',   type=int, required=True)
    parser.add_argument('--start_month',  type=int, default=1)
    parser.add_argument('--end_year',     type=int, required=True)
    parser.add_argument('--end_month',    type=int, default=12)
    parser.add_argument('--period',       type=int, required=True,
                        help='window length in months')
    parser.add_argument('--epochs',       type=int, default=100)
    parser.add_argument('--lr',           type=float, default=0.005)
    parser.add_argument('--train_ratio',  type=float, default=0.7)
    parser.add_argument('--val_ratio',    type=float, default=0.15)
    parser.add_argument('--neg_ratio',    type=float, default=1.0,
                        help='ratio of negative to positive edges')
    parser.add_argument('--alpha',        type=float, default=0.5,
                        help='weight for feature vs edge loss')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows = parse_windows(
        args.start_year, args.start_month,
        args.end_year, args.end_month,
        args.period
    )
    # consecutive pairs
    pairs = list(zip(windows, windows[1:]))
    print(f"Training on window pairs: {pairs}")

    # infer dims from first window
    w0, w1 = pairs[0]
    gfn = f"graph_{w0[0]}-{w0[1]:02d}_{w0[2]}-{w0[3]:02d}_filtered.json"
    fx0 = f"features_{w0[0]}-{w0[1]:02d}_{w0[2]}-{w0[3]:02d}_filtered.npy"
    g0 = json.load(open(os.path.join(BASEPATH, gfn)))
    x0 = np.load(os.path.join(BASEPATH, fx0))
    num_nodes, in_dim = x0.shape

    # model
    edge_latent_dim = 2
    encoder = EdgeEncoder(in_channels=in_dim, hidden_channels=64, edge_latent_dim=edge_latent_dim).to(device)
    edge_decoder = EdgeDecoder(edge_latent_dim=edge_latent_dim).to(device)
    feat_decoder = FeatureDecoder(edge_latent_dim=edge_latent_dim, out_features=in_dim).to(device)

    params = list(encoder.parameters()) + \
             list(edge_decoder.parameters()) + \
             list(feat_decoder.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    crit_edge = torch.nn.BCELoss()
    crit_feat = torch.nn.MSELoss()

    best_val = float('inf')
    best_state = None

    for epoch in range(1, args.epochs+1):
        encoder.train()
        edge_decoder.train()
        feat_decoder.train()
        train_loss = 0.0
        val_loss = 0.0

        edge_losses, feat_losses = [], []
        val_edge_losses, val_feat_losses = [], []

        train_preds, train_labels = [], []
        val_preds, val_labels = [], []

        for (y0, m0, y1, m1), (yn0, mn0, yn1, mn1) in pairs:
            # load current graph
            gfn = f"graph_{y0}-{m0:02d}_{y1}-{m1:02d}_filtered.json"
            g = json.load(open(os.path.join(BASEPATH, gfn)))
            ei_pos = torch.tensor(g['edge_index'], dtype=torch.long, device=device)
            pos_list = list(zip(ei_pos[0].tolist(), ei_pos[1].tolist()))
            pos_set = set(pos_list)
            n_pos = ei_pos.size(1)

            # load features
            fx_fn = f"features_{y0}-{m0:02d}_{y1}-{m1:02d}_filtered.npy"
            fx_next = f"features_{yn0}-{mn0:02d}_{yn1}-{mn1:02d}_filtered.npy"
            x_t = torch.tensor(
                np.load(os.path.join(BASEPATH, fx_fn)),
                dtype=torch.float,
                device=device
            )
            x_tp1 = torch.tensor(
                np.load(os.path.join(BASEPATH, fx_next)),
                dtype=torch.float,
                device=device
            )

            # encode nodes once
            h = F.relu(encoder.gcn1(x_t, ei_pos, None))
            h = F.relu(encoder.gcn2(h, ei_pos, None))

            # positive edge embeddings
            z_pos = encoder.edge_mlp(
                torch.cat([h[ei_pos[0]], h[ei_pos[1]]], dim=1)
            )

            # negative sampling
            ei_neg = sample_negative_edges(
                num_nodes, pos_set,
                int(n_pos * args.neg_ratio),
                device
            )
            z_neg = encoder.edge_mlp(
                torch.cat([h[ei_neg[0]], h[ei_neg[1]]], dim=1)
            )

            # combine
            z_all = torch.cat([z_pos, z_neg], dim=0)
            labels = torch.cat([
                torch.ones(n_pos, device=device),
                torch.zeros(z_neg.size(0), device=device)
            ], dim=0)

            # predict edges
            preds = edge_decoder(z_all)

            # shuffle + split
            perm = torch.randperm(preds.size(0), device=device)
            preds, labels = preds[perm], labels[perm]
            tr_mask, va_mask, _ = split_masks(
                preds.size(0), args.train_ratio, args.val_ratio, device=device
            )

            # Collect predictions and labels for ROC-AUC
            train_preds.append(preds[tr_mask].detach().cpu().numpy())
            train_labels.append(labels[tr_mask].detach().cpu().numpy())
            val_preds.append(preds[va_mask].detach().cpu().numpy())
            val_labels.append(labels[va_mask].detach().cpu().numpy())

            edge_loss = crit_edge(preds[tr_mask], labels[tr_mask])
            edge_losses.append(edge_loss)

            # feature reconstruction
            feat_pred = feat_decoder(z_pos, ei_pos, num_nodes)
            feat_loss = crit_feat(feat_pred, x_tp1)
            feat_losses.append(feat_loss)

            # combined
            loss = args.alpha * feat_loss + (1 - args.alpha) * edge_loss
            loss.backward()
            train_loss += loss.item()

            # validation
            with torch.no_grad():
                edge_val = crit_edge(preds[va_mask], labels[va_mask])
                feat_val = crit_feat(feat_pred, x_tp1)
                val_loss += args.alpha * feat_val + (1 - args.alpha) * edge_val
            val_edge_losses.append(edge_val)
            val_feat_losses.append(feat_val)

        optimizer.step()
        optimizer.zero_grad()

        avg_tr = train_loss / len(pairs)
        avg_va = val_loss   / len(pairs)

        # Compute ROC-AUC for train and validation
        train_preds = np.concatenate(train_preds)
        train_labels = np.concatenate(train_labels)
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)

        train_auc = roc_auc_score(train_labels, train_preds)
        val_auc = roc_auc_score(val_labels, val_preds)

        if avg_va < best_val:
            best_val = avg_va
            best_state = {
                'enc': encoder.state_dict(),
                'ed': edge_decoder.state_dict(),
                'fd': feat_decoder.state_dict()
            }
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch}: Train Loss={avg_tr:.4f} "
                f"(Edge {np.mean([e.cpu().item() for e in edge_losses]):.4f} "
                f"Feat {np.mean([f.cpu().item() for f in feat_losses]):.4f}) | "
                f"Val Loss={avg_va:.4f} "
                f"(Edge {np.mean([ve.cpu().item() for ve in val_edge_losses]):.4f} "
                f"Feat {np.mean([vf.cpu().item() for vf in val_feat_losses]):.4f}) "
                f"Train AUC={train_auc:.4f} | Val AUC={val_auc:.4f}"
            )

        edge_losses, feat_losses = [], []
        val_edge_losses, val_feat_losses = [], []

    # restore & save best
    if best_state:
        encoder.load_state_dict(best_state['enc'])
        edge_decoder.load_state_dict(best_state['ed'])
        feat_decoder.load_state_dict(best_state['fd'])
        torch.save(best_state, os.path.join(BASEPATH, 'best_model.pth'))
        print(f"Best model saved, Val Loss={best_val:.4f}")

if __name__ == '__main__':
    main()
