# Lead project author and repository maintainer: Muhammad Abdullah Anwar
"""Graph convolutional surrogate model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv

class IEEE9_GNN(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=100, out_channels=4):
        super(IEEE9_GNN, self).__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv_out = GraphConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv_out(x, edge_index)  # [num_nodes_total, 4]
        return x

def MSE(yhat, y):
    return torch.mean((yhat - y) ** 2)

