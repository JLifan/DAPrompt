import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, SAGEConv, GATConv, GCNConv, TransformerConv
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix, degree


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
    
    def forward_weighted(self, x, edge_index, weight):
        x = F.relu(self.conv1(x, edge_index, weight))
        x = self.conv2(x, edge_index, weight)
        return x

    def forward_subgraph(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.linear(x)
        x = global_mean_pool(x, batch)
        return x
    
    def forward_cl(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return x


class GATEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.linear = nn.Linear(hidden_dim, out_dim) 

    def forward(self, x, edge_index):
        x = F.relu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x

    def forward_subgraph(self, x, edge_index, batch):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.linear(x)
        x = global_mean_pool(x, batch)
        return x


class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim) 

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

    def forward_subgraph(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.linear(x)
        x = global_mean_pool(x, batch)
        return x


class GraphTransformerEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4):
        super().__init__()
        self.conv1 = TransformerConv(in_dim, hidden_dim, heads=heads)
        self.conv2 = TransformerConv(hidden_dim * heads, hidden_dim, heads=1)
        self.linear = nn.Linear(hidden_dim, out_dim) 
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

    def forward_subgraph(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.linear(x)
        x = global_mean_pool(x, batch)
        return x


def edge_index_to_sparse_tensor_adj(edge_index, num_nodes):
    sparse_adj_adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    values = sparse_adj_adj.data
    indices = np.vstack((sparse_adj_adj.row, sparse_adj_adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = sparse_adj_adj.shape
    sparse_adj_adj_tensor = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    return sparse_adj_adj_tensor

def gcn_norm(edge_index, num_nodes, device):
    a1 = edge_index_to_sparse_tensor_adj(edge_index, num_nodes).to(device)
    d1_adj = torch.diag(degree(edge_index[0], num_nodes=num_nodes)).to_sparse()
    d1_adj = torch.pow(d1_adj, -0.5)
    return torch.sparse.mm(torch.sparse.mm(d1_adj, a1), d1_adj)

class GeomGCN_layer(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim * 4, out_dim)

    def forward(self, x, edge_index, edge_relation, norm_adj, num_nodes, device):
        relation_adjs = []
        for i in range(4):
            mask = (edge_relation == i)
            edge_index_i = edge_index[:, mask]
            relation_adj = edge_index_to_sparse_tensor_adj(edge_index_i, num_nodes).to(device)
            relation_adjs.append(relation_adj)
        h_all = []
        for adj in relation_adjs:
            h = torch.sparse.mm(torch.mul(adj, norm_adj), x)
            h_all.append(h)
        h = torch.cat(h_all, dim=1)
        return self.linear(h)

class GeomGCNEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, device):
        super().__init__()
        self.device = device
        self.gcn1 = GeomGCN_layer(in_dim, hidden_dim)
        self.gcn2 = GeomGCN_layer(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim) 

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        edge_relation = torch.zeros(num_edges, dtype=torch.long, device=self.device)
        norm_adj = gcn_norm(edge_index, num_nodes, self.device)
        h = self.gcn1(x, edge_index, edge_relation, norm_adj, num_nodes, self.device)
        h = F.relu(h)
        h = self.gcn2(h, edge_index, edge_relation, norm_adj, num_nodes, self.device)
        return h

    def forward_subgraph(self, x, edge_index, batch):
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        edge_relation = torch.zeros(num_edges, dtype=torch.long, device=self.device)
        norm_adj = gcn_norm(edge_index, num_nodes, self.device)
        h = self.gcn1(x, edge_index, edge_relation, norm_adj, num_nodes, self.device)
        h = F.relu(h)
        h = self.gcn2(h, edge_index, edge_relation, norm_adj, num_nodes, self.device)
        h = F.relu(h)
        x = self.linear(h)
        x = global_mean_pool(x, batch)
        return x












