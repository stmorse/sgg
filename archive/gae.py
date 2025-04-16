# simple_edge_gae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean          # ships with PyG
from torch_geometric.data import Data, DataLoader

class EdgeGAE(nn.Module):
    """
    Graph auto‑encoder whose *latent variables live on edges*.
    Enc ↦ node → edge  latent  Z_e
    Dec ↦ edge → node  recon   X̂
    """
    def __init__(self, in_dim, hid_dim, z_dim, out_dim):
        super().__init__()
        # encoder: node → hidden h
        self.gcn1 = GCNConv(in_dim, hid_dim)
        self.gcn2 = GCNConv(hid_dim, hid_dim)

        # edge‑MLP: [h_u || h_v] → z_e
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*hid_dim, z_dim),
            nn.ReLU(),
        )

        # decoder: aggregated edge latents → x̂_i
        self.node_mlp = nn.Sequential(
            nn.Linear(z_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index          # (N,F), (2,E)

        # ---- encoder (node → hidden) ----
        h = F.relu(self.gcn1(x, edge_index))
        h = self.gcn2(h, edge_index)                     # (N,hid)

        # ---- build edge latent Z_e ----
        src, dst = edge_index                            # (E)
        h_cat = torch.cat([h[src], h[dst]], dim=1)       # (E,2*hid)
        z_e = self.edge_mlp(h_cat)                       # (E,z_dim)

        # ---- aggregate edge latents back to nodes ----
        # use mean of incident edge latents; could choose sum / max etc.
        z_node = scatter_mean(z_e, src, dim=0, dim_size=h.size(0)) + \
                 scatter_mean(z_e, dst, dim=0, dim_size=h.size(0))
        z_node = z_node / 2                              # symmetric mean

        # ---- decoder (node ← latent) ----
        x_hat = self.node_mlp(z_node)                    # (N,out_dim)

        return x_hat, z_e

# ----------------------- usage skeleton -----------------------

def train(model, loader, epochs=200, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs+1):
        model.train()
        tot_loss = 0
        for data in loader:
            data = data.to(next(model.parameters()).device)
            x_hat, _ = model(data)
            loss = F.mse_loss(x_hat, data.x)             # ‖X − X̂‖²
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()*data.num_nodes
        if epoch % 20 == 0:
            print(f"epoch {epoch:3d} | loss {tot_loss/len(loader.dataset):.4f}")

# ----------------------- example -----------------------

if __name__ == "__main__":
    # toy graph: 4 nodes, undirected edges, 3‑dim node features
    x  = torch.randn(4, 3)
    edge_index = torch.tensor([[0,1,2,3,0,2],
                               [1,0,3,2,2,0]])          # (2,E)
    data = Data(x=x, edge_index=edge_index)
    loader = DataLoader([data], batch_size=1)

    model = EdgeGAE(in_dim=3, hid_dim=16, z_dim=2, out_dim=3)
    train(model, loader, epochs=100)
    # Print the learned edge latent vectors after training
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(next(model.parameters()).device)
            _, z_e = model(data)
            print("Learned edge latent vectors (z_e):")
            print(z_e)
