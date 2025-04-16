import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

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

if __name__ == "__main__":
    # Quick test run: dummy data
    num_nodes = 10
    in_channels = 16
    out_channels = 16
    dummy_x = torch.randn(num_nodes, in_channels)
    dummy_edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
                                     [1, 2, 3, 4, 5, 6]], dtype=torch.long)
    dummy_edge_weight = torch.ones(dummy_edge_index.size(1))
    from torch_geometric.data import Data
    dummy_data = Data(x=dummy_x, edge_index=dummy_edge_index, edge_weight=dummy_edge_weight)
    
    model = GNN(in_channels, out_channels)
    output = model(dummy_data)
    print("Output shape:", output.shape)
