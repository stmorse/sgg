#!/usr/bin/env python3
import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data

from models import EdgeEncoder, EdgeDecoder, FeatureDecoder

BASEPATH = '/sciclone/geograd/stmorse/reddit/subreddit/science/filtered'

def sample_negative_edges(num_nodes, pos_set, num_samples, device):
    neg = set()
    while len(neg) < num_samples:
        # sample in batches
        idx = torch.randint(0, num_nodes, (num_samples * 2, 2), device=device)
        for u, v in idx.tolist():
            if u == v:
                continue
            pair = (u, v)
            if pair in pos_set or (v, u) in pos_set:
                continue
            neg.add(pair)
            if len(neg) >= num_samples:
                break
    neg = list(neg)
    src = torch.tensor([u for u, v in neg], dtype=torch.long, device=device)
    dst = torch.tensor([v for u, v in neg], dtype=torch.long, device=device)
    return torch.stack([src, dst], dim=0)


def split_masks(num_examples, train_ratio, val_ratio, seed=42, device=None):
    torch.manual_seed(seed)
    idx = torch.randperm(num_examples, device=device)
    train_end = int(train_ratio * num_examples)
    val_end = train_end + int(val_ratio * num_examples)
    mask = torch.zeros(num_examples, dtype=torch.bool, device=device)
    train_mask = mask.clone()
    val_mask = mask.clone()
    test_mask = mask.clone()
    train_mask[idx[:train_end]] = True
    val_mask[idx[train_end:val_end]] = True
    test_mask[idx[val_end:]] = True
    return train_mask, val_mask, test_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train edge-level GAE with negative sampling and combined loss")
    parser.add_argument('--start_year', type=int, required=True)
    parser.add_argument('--end_year', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--neg_ratio', type=float, default=1.0,
                        help='ratio of negative to positive edges')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='weight for feature loss vs edge loss: loss = alpha*feat + (1-alpha)*edge')
    args = parser.parse_args()

    years = list(range(args.start_year, args.end_year))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on year pairs: {[(y, y+1) for y in years]}")

    # Initialize model
    # load one sample to get dims
    with open(os.path.join(BASEPATH,f"graph_{years[0]}_union.json")) as f:
        g0 = json.load(f)
    x0 = np.load(os.path.join(BASEPATH, f"features_{years[0]}_union.npy"))
    num_nodes, in_dim = x0.shape

    encoder = EdgeEncoder(in_channels=in_dim, hidden_channels=64, edge_latent_dim=32).to(device)
    edge_decoder = EdgeDecoder(edge_latent_dim=32).to(device)
    feature_decoder = FeatureDecoder(edge_latent_dim=32, out_features=in_dim).to(device)

    params = list(encoder.parameters()) + list(edge_decoder.parameters()) + list(feature_decoder.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    criterion_edge = nn.BCELoss()
    criterion_feat = nn.MSELoss()

    best_val = float('inf')
    best_state = None

    for epoch in range(1, args.epochs+1):
        encoder.train(); edge_decoder.train(); feature_decoder.train()
        total_train_loss = 0.0
        total_val_loss = 0.0

        for y in years:
            # load data for time t and t+1
            with open(os.path.join(BASEPATH,f"graph_{years[0]}_union.json")) as f:
                g = json.load(f)
            edge_index_pos = torch.tensor(g['edge_index'], dtype=torch.long, device=device)
            pos_list = list(zip(edge_index_pos[0].tolist(), edge_index_pos[1].tolist()))
            pos_set = set(pos_list)
            num_pos = edge_index_pos.size(1)

            x_t = torch.tensor(
                np.load(os.path.join(BASEPATH, f"features_{y}_union.npy")), 
                dtype=torch.float, device=device)
            x_tp1 = torch.tensor(
                np.load(os.path.join(BASEPATH, f"features_{y+1}_union.npy")), 
                dtype=torch.float, device=device)

            # encode node embeddings via GCN layers
            h = F.relu(encoder.gcn1(x_t, edge_index_pos, None))
            h = F.relu(encoder.gcn2(h, edge_index_pos, None))

            # positive edge embeddings
            z_pos = encoder.edge_mlp(torch.cat([h[edge_index_pos[0]], h[edge_index_pos[1]]], dim=1))

            # sample negative edges
            edge_neg = sample_negative_edges(num_nodes, pos_set,
                                             int(num_pos * args.neg_ratio), device)
            z_neg = encoder.edge_mlp(torch.cat([h[edge_neg[0]], h[edge_neg[1]]], dim=1))

            # combine pos & neg
            z_all = torch.cat([z_pos, z_neg], dim=0)
            labels_all = torch.cat([torch.ones(num_pos, device=device),
                                     torch.zeros(z_neg.size(0), device=device)], dim=0)

            # decode edges
            preds_all = edge_decoder(z_all)

            # shuffle and split
            perm = torch.randperm(preds_all.size(0), device=device)
            preds_all = preds_all[perm]
            labels_all = labels_all[perm]
            train_mask, val_mask, _ = split_masks(preds_all.size(0),
                                                  args.train_ratio, args.val_ratio,
                                                  device=device)

            edge_loss = criterion_edge(preds_all[train_mask], labels_all[train_mask])

            # feature reconstruction (uses only true edges for aggregation)
            feat_preds = feature_decoder(z_pos, edge_index_pos, num_nodes)
            feat_loss = criterion_feat(feat_preds, x_tp1)

            # combined loss
            loss = args.alpha * feat_loss + (1 - args.alpha) * edge_loss
            loss.backward()
            total_train_loss += loss.item()

            # validation loss
            with torch.no_grad():
                val_edge_loss = criterion_edge(preds_all[val_mask], labels_all[val_mask])
                val_feat_loss = criterion_feat(feat_preds, x_tp1)
                val_loss = args.alpha * val_feat_loss + (1 - args.alpha) * val_edge_loss
                total_val_loss += val_loss.item()

            print(
                f' > {y}: (TNG) edge {edge_loss:.4f} feat {feat_loss:.4f} | '
                f' (VAL) edge {val_edge_loss:.4f} feat {val_feat_loss:.4f}'
            )

        optimizer.step()
        optimizer.zero_grad()

        avg_train = total_train_loss / len(years)
        avg_val = total_val_loss / len(years)
        if avg_val < best_val:
            best_val = avg_val
            best_state = {
                'enc': encoder.state_dict(),
                'edg_dec': edge_decoder.state_dict(),
                'feat_dec': feature_decoder.state_dict()
            }

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch}: Avg Train Loss={avg_train:.4f}, Avg Val Loss={avg_val:.4f}")

    # load best
    if best_state:
        encoder.load_state_dict(best_state['enc'])
        edge_decoder.load_state_dict(best_state['edg_dec'])
        feature_decoder.load_state_dict(best_state['feat_dec'])

    # Save the best state to a file
    save_path = os.path.join(BASEPATH, "best_model.pth")
    torch.save(best_state, save_path)
    print(f"Best model saved to {save_path}")

    print("Training complete.")
