import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# ----
# GAE (double decoder)
# ----

class EdgeEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_latent_dim):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        # MLP to produce edge embeddings from node embeddings
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, edge_latent_dim)
        )

    def forward(self, x, edge_index, edge_weight=None):
        # Node embeddings via GCN
        h = F.relu(self.gcn1(x, edge_index, edge_weight))
        h = F.relu(self.gcn2(h, edge_index, edge_weight))

        # Form edge embeddings
        src, dst = edge_index
        h_src = h[src]
        h_dst = h[dst]

        edge_input = torch.cat([h_src, h_dst], dim=1)
        z_uv = self.edge_mlp(edge_input)

        return z_uv

class EdgeDecoder(nn.Module):
    """
    Decodes edge embeddings z_uv back to edge probabilities.
    """
    def __init__(self, edge_latent_dim):
        super().__init__()
        self.lin = nn.Linear(edge_latent_dim, 1)

    def forward(self, z_uv):
        logits = self.lin(z_uv).squeeze(-1)
        return torch.sigmoid(logits)

class FeatureDecoder(nn.Module):
    """
    Decodes node-level features from aggregated edge embeddings.
    Aggregation: Mean pooling over incident edges.
    """
    def __init__(self, edge_latent_dim, out_features):
        super().__init__()
        self.lin = nn.Linear(edge_latent_dim, out_features)

    def forward(self, z_uv, edge_index, num_nodes):
        # Aggregate edge embeddings back to node embeddings
        src, dst = edge_index
        agg = torch.zeros((num_nodes, z_uv.size(1)), device=z_uv.device)

        # Sum edge embeddings to source and destination nodes
        agg.index_add_(0, src, z_uv)
        agg.index_add_(0, dst, z_uv)

        # Count edges per node (for mean pooling)
        deg = torch.bincount(torch.cat([src, dst]), minlength=num_nodes).unsqueeze(-1).clamp(min=1)
        agg = agg / deg  # mean pooling

        # Decode aggregated embeddings to features
        return self.lin(agg)



# ----
# Simple GNNs
# ----

class GNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, dropout=0.5):
        super().__init__()
        # First two layers perform message passing over the weighted graph.
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Pass edge_weight if available.
        edge_weight = getattr(data, "edge_weight", None)
        
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final prediction layer maps to t1 feature space.
        x = self.fc(x)
        return x

class GCNFeat(torch.nn.Module):
    """Simple 2-layer GCN for baseline testing"""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
    
    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)

class SimpleGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv = GCNConv(in_dim, hidden_dim)
        
    def forward(self, x, edge_index):
        h = F.relu(self.conv(x, edge_index))
        return h

if __name__ == "__main__":
    print('Testing dummy data...')

    num_nodes, in_features, hidden_dim, edge_dim, out_features = 100, 15, 32, 16, 15

    x = torch.randn(num_nodes, in_features)
    edge_index = torch.randint(0, num_nodes, (2, 500))

    encoder = EdgeEncoder(in_features, hidden_dim, edge_dim)
    edge_decoder = EdgeDecoder(edge_dim)
    feature_decoder = FeatureDecoder(edge_dim, out_features)

    z_uv = encoder(x, edge_index)
    reconstructed_edges = edge_decoder(z_uv)
    reconstructed_features = feature_decoder(z_uv, edge_index, num_nodes)

    print(reconstructed_edges.shape)     # torch.Size([num_edges])
    print(reconstructed_features.shape)  # torch.Size([num_nodes, out_features])
 

    # -----
    # for GNN:
    # -----
    # num_nodes = 10
    # in_channels = 16
    # out_channels = 16
    # dummy_x = torch.randn(num_nodes, in_channels)
    # dummy_edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
    #                                  [1, 2, 3, 4, 5, 6]], dtype=torch.long)
    # dummy_edge_weight = torch.ones(dummy_edge_index.size(1))
    
    # print('Creating data object...')
    # from torch_geometric.data import Data
    # dummy_data = Data(x=dummy_x, edge_index=dummy_edge_index, edge_weight=dummy_edge_weight)
    
    # print('Forward pass of GNN...')
    # model = GNN(in_channels, out_channels)
    # output = model(dummy_data)
    # print("Output shape:", output.shape)
