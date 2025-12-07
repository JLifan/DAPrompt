from torch_geometric.utils import to_undirected, add_self_loops
from collections import defaultdict
import random
import numpy as np
from torch_geometric.data import Data, Dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix


class DAPrompt(nn.Module):
    def __init__(
            self, 
            encoder, 
            structure_token_number, 
            semantics_token_number, 
            input_dim, 
            hid_dim, 
            device, 
            outer_thre,
            lamda1,
            lamda2,
            structure_threshold,
            broadcast
            ):
        super(DAPrompt, self).__init__()
        self.encoder = encoder  # this is the backbone; the default is a 2-layer GCN
        self.structure_token_number = structure_token_number  # number of structure prompt tokens
        self.semantics_token_number = semantics_token_number  # number of semantics prompt tokens
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.device = device
        self.outer_thre = outer_thre
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.broadcast = broadcast
        self.structure_threshold = structure_threshold
        # structure prompt
        self.structure_prompt = nn.Parameter(torch.empty(self.structure_token_number, self.input_dim).to(self.device))
        nn.init.xavier_uniform_(self.structure_prompt)
        # semantics prompt
        self.semantics_prompt = nn.Parameter(torch.empty(self.semantics_token_number, self.input_dim).to(self.device))
        nn.init.xavier_uniform_(self.semantics_prompt)

    def inner_edge(self, X, thre, type='cos'):
        if type == 'product':
            similarity_matrix = torch.mm(X, X.t())
            similarity_matrix = torch.sigmoid(similarity_matrix)
            adj_matrix = (similarity_matrix > thre).float()
            edge_index = adj_matrix.nonzero().t()
        elif type == 'cos':
            # this is the default
            X_norm = F.normalize(X, p=2, dim=1)
            similarity_matrix = torch.mm(X_norm, X_norm.t())
            similarity_matrix = torch.sigmoid(similarity_matrix)
            adj_matrix = (similarity_matrix > thre).float()
            edge_index = adj_matrix.nonzero().t()
        elif type == 'Euclidean':
            dist = torch.cdist(X, X, p=2)
            similarity_matrix = -dist
            similarity_matrix = torch.sigmoid(similarity_matrix)
            adj_matrix = (similarity_matrix > thre).float()
            edge_index = adj_matrix.nonzero().t()
        return edge_index

    def dedup_edges_by_adjacency(self, subgraph_edge_index, all_node_edge_index, num_nodes, threshold):
        adj1 = to_scipy_sparse_matrix(subgraph_edge_index, num_nodes=num_nodes).tocoo()
        adj2 = to_scipy_sparse_matrix(all_node_edge_index, num_nodes=num_nodes).tocoo()
        adj_sum = (adj1 + adj2).astype(float).tocoo()
        values = torch.tensor(adj_sum.data, dtype=torch.float)
        sigmoid_values = torch.sigmoid(values)
        mask = sigmoid_values > threshold
        row = torch.tensor(adj_sum.row)[mask]
        col = torch.tensor(adj_sum.col)[mask]
        edge_index = torch.stack([row, col], dim=0)
        return edge_index

    def create_manipulated_graph(self, batch: Batch):
        subgraphs = []
        for subgraph in Batch.to_data_list(batch):
            # New feature matrix, X, is constructed by concatenating three parts in order:
            # 1. structure prompt
            # 2. original node features of the subgraph
            # 3. semantic prompt
            subgraph_edge_index = subgraph.edge_index + self.structure_token_number
            subgraph_X = subgraph.x
            new_X = torch.cat((self.structure_prompt, subgraph_X), dim=0)
            all_node_edge_index = self.inner_edge(new_X, self.outer_thre)
            normal_nodes_number = subgraph.x.shape[0] + self.structure_token_number
            # Interaction between the target node and semantics tokens
            target_edge_self = torch.tensor([[subgraph.center_id + self.structure_token_number, normal_nodes_number+i] for i in range(self.semantics_token_number)]).t()
            target_edge_self = target_edge_self.to(self.device)
            target_edge_self = to_undirected(target_edge_self) 
            # Find an optimal semantics token, optionally
            connect = []
            for i in range(self.semantics_token_number):
                cos_sim = F.cosine_similarity(subgraph.x[subgraph.center_id].squeeze(), self.semantics_prompt[i], dim=0)
                sim = torch.sigmoid(cos_sim)
                connect.append(sim)
            max_index = connect.index(max(connect))
            result = [i == max_index for i in range(len(connect))]
            connect = result
            # Semantics tokens interact with the remaining nodes
            target_edge = []
            for i, j in zip(range(self.semantics_token_number), connect):
                if True or j:  # 'if j' represents only using the optimal semantics token
                    if self.broadcast:
                        for node_id in range(self.structure_token_number, normal_nodes_number):
                            target_edge.append([normal_nodes_number+i, node_id])
            target_edge = torch.tensor(target_edge).to(self.device).t()
            # structure prompt
            tatal_nodes_number = normal_nodes_number + self.semantics_token_number
            concat_edge_index = self.dedup_edges_by_adjacency(subgraph_edge_index, all_node_edge_index, tatal_nodes_number, self.structure_threshold).to(self.device)
            new_edge_index = torch.cat((concat_edge_index, target_edge, target_edge_self), dim=1)
            new_X = torch.cat((new_X, self.semantics_prompt), dim=0)
            manipulated_graph = Data(x=new_X, edge_index=new_edge_index)
            subgraphs.append(manipulated_graph)
        new_batch = Batch.from_data_list(subgraphs)
        return new_batch

    def forward(self, batch):
        new_batch = self.create_manipulated_graph(batch)
        out = self.encoder.forward_subgraph(new_batch.x, new_batch.edge_index, new_batch.batch)
        return out

    def loss(self, batch):
        out = self.forward(batch)
        loss = F.cross_entropy(out, batch.y)
        # loss_prompt = self.prompt_similarity_loss(self.prompt)
        loss_prompt = self.prompt_diversity_loss(self.structure_prompt)
        loss_target_tokens = self.prompt_diversity_loss(self.semantics_prompt)
        loss = loss + self.lamda1*loss_prompt + self.lamda2*loss_target_tokens
        return loss

    def prompt_diversity_loss(self, prompt):
        prompt = F.normalize(prompt, p=2, dim=1)
        sim_matrix = torch.matmul(prompt, prompt.T)
        N = sim_matrix.size(0)
        mask = torch.eye(N, dtype=torch.bool, device=prompt.device)
        loss = sim_matrix[~mask].mean()
        return loss

    def prompt_similarity_loss(self, prompt):
        prompt = F.normalize(prompt, p=2, dim=1)
        sim_matrix = torch.matmul(prompt, prompt.T)
        N = sim_matrix.size(0)
        mask = torch.eye(N, dtype=torch.bool, device=prompt.device)
        loss = -sim_matrix[~mask].mean()
        return loss

    def prompt_diversity_loss_var(self, prompt):
        return -torch.var(prompt, dim=0).mean()

    def prompt_diversity_loss_pairwise_distance(self, prompt):
        N = prompt.size(0)
        dist_sum = 0
        count = 0
        for i in range(N):
            for j in range(i + 1, N):
                dist = F.pairwise_distance(prompt[i].unsqueeze(0), prompt[j].unsqueeze(0), p=2)
                dist_sum += dist
                count += 1
        return -dist_sum / count


