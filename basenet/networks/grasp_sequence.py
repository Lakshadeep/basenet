import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import HeteroData, Data
from torch_geometric.datasets import Planetoid, AQSOL, DBLP, IMDB
from torch_geometric.utils import to_networkx
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, SAGEConv, to_hetero, GATv2Conv, HeteroConv, Linear
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData, Data


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def reset_parameters(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_uniform_(layer.weight, a=1.0, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PolicyNet(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GATv2Conv((-1, -1), hidden_dim, add_self_loops=False)
        self.conv2 = GATv2Conv((-1, -1), hidden_dim, add_self_loops=False)
        # self.conv3 = GATv2Conv((-1, -1), hidden_dim, add_self_loops=False)
        # self.conv4 = GATv2Conv((-1, -1), hidden_dim, add_self_loops=False)
        self.conv5 = GATv2Conv((-1, -1), hidden_dim, add_self_loops=False)
        self.mlp = MLP(hidden_dim, hidden_dim, output_dim) #.to(device='cuda:0')

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        # x = self.conv3(x, edge_index).relu()
        # x = self.conv4(x, edge_index).relu()
        x = self.conv5(x, edge_index)
        x = self.mlp(x)
        return x

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('robot', 'grasps', 'object'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ('object', 'grasps', 'robot'): SAGEConv(-1, hidden_channels, add_self_loops=False),
                ('object', 'loops', 'object'): SAGEConv((-1, -1), hidden_channels, add_self_loops=False),
                ('object', 'surrounds', 'object'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict['object'])
