import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    """
    Baseline Model:
    - A simple two-layer GCN model, similar to https://github.com/tkipf/pygcn
    - Implement with DGL package
    """
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
    
        self.layers.append(GraphConv(in_size, hid_size, activation=F.relu))
        self.layers.append(GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, g, features):
        h = features
        h = self.layers[0](g, h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.layers[1](g, h)
        h = F.log_softmax(h, dim=1)
        return h

# class YourGNNModel(nn.Module):
#     """
#     TODO: Use GCN model as reference, implement your own model here to achieve higher accuracy on testing data
#     """
#     def __init__(self, in_size, hid_size, out_size):
#         super().__init__()
    
#     def forward(self, g, features):
#         pass
