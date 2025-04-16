import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data

from models import GNN  # your PyG model in models.py

def split_masks(num_nodes, train_ratio, val_ratio, seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(num_nodes)
    train_end = int(train_ratio * num_nodes)
    val_end = train_end + int(val_ratio * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[idx[:train_end]] = True
    val_mask[idx[train_end:val_end]] = True
    test_mask[idx[val_end:]] = True
    return train_mask, val_mask, test_mask

def train_epoch(model, data, mask, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, mask, criterion):
    model.eval()
    with torch.no_grad():
        out = model(data)
    return criterion(out[mask], data.y[mask]).item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_json", type=str, required=True,
                        help="Path to graph JSON (from previous script)")
    parser.add_argument("--features_t0", type=str, required=True,
                        help="Path to .npy file with t0 features")
    parser.add_argument("--features_t1", type=str, required=True,
                        help="Path to .npy file with t1 features")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Fraction of nodes for training")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Fraction for validation")
    args = parser.parse_args()

    # Load graph JSON: expects "user_to_idx", "edge_index", "edge_weight"
    with open(args.graph_json, "r") as f:
        graph = json.load(f)
    user_to_idx = graph["user_to_idx"]
    num_nodes = len(user_to_idx)
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)  # shape [2, E]
    edge_weight = torch.tensor(graph["edge_weight"], dtype=torch.float)

    # Load node features (t0 and t1); rows must match user_to_idx order.
    feats_t0 = np.load(args.features_t0)  # shape (num_nodes, k)
    feats_t1 = np.load(args.features_t1)  # shape (num_nodes, k)
    assert feats_t0.shape[0] == num_nodes and feats_t1.shape[0] == num_nodes, "Mismatch in node count."

    x = torch.tensor(feats_t0, dtype=torch.float)
    y = torch.tensor(feats_t1, dtype=torch.float)

    data = Data(x=x, y=y, edge_index=edge_index, edge_weight=edge_weight)

    train_mask, val_mask, test_mask = split_masks(num_nodes, args.train_ratio, args.val_ratio)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    model = GNN(in_channels=x.size(1), out_channels=y.size(1)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        t_loss = train_epoch(model, data, train_mask, optimizer, criterion)
        v_loss = evaluate(model, data, val_mask, criterion)
        if v_loss < best_val:
            best_val = v_loss
            best_state = model.state_dict()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}: Train Loss {t_loss:.4f}, Val Loss {v_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    t_loss = evaluate(model, data, test_mask, criterion)
    print(f"Test Loss: {t_loss:.4f}")

if __name__ == "__main__":
    main()
