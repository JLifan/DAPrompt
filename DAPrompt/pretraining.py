from torch_geometric.utils import to_dense_adj
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class StructureEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, threshold=0.5):
        super(StructureEncoder, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, out_features)
        self.threshold = threshold

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        similarity_matrix = self.calculate_similarity(x)
        edge_index = self.get_edges_from_similarity(similarity_matrix)
        return edge_index

    def calculate_similarity(self, x):
        """
        Compute the cosine similarity between node embeddings.

        :param x: Node embeddings (shape: [num_nodes, embedding_dim])
        :return: Similarity matrix (shape: [num_nodes, num_nodes])
        """
        similarity_matrix = torch.matmul(x, x.t())
        norm_x = torch.norm(x, p=2, dim=1, keepdim=True)
        similarity_matrix = similarity_matrix / (norm_x * norm_x.t())
        return similarity_matrix

    def get_edges_from_similarity(self, similarity_matrix):
        """
        Generate edges based on the similarity matrix.

        :param similarity_matrix: Pairwise similarity matrix between nodes
        :return: edge_index generated from the similarity matrix
        """
        edge_index = (similarity_matrix > self.threshold).nonzero().t()
        return edge_index

    def generate_A(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        similarity_matrix = self.calculate_similarity(x)
        return similarity_matrix


class Structure_pretrain(nn.Module):
    def __init__(self, Encoder, number_nodes, tau, data, threshold, device):
        super().__init__()
        self.encoder = Encoder  # GCN
        self.number_nodes = number_nodes
        self.tau = tau  
        self.threshold = threshold  
        self.device = device
        self.data = data.to(device)
        self.A = to_dense_adj(self.data.edge_index)[0]

        self.structure_learner = StructureEncoder(
            in_features=self.data.x.size(1),
            hidden_features=32,  
            out_features=32,  
            threshold=threshold  
        ).to(device)
        self._initialize_weights()

    def _initialize_weights(self):
        if isinstance(self.encoder, nn.Module):
            for name, param in self.encoder.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)
                elif 'bias' in name:
                    init.zeros_(param)
        if isinstance(self.structure_learner, nn.Module):
            for name, param in self.structure_learner.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)
                elif 'bias' in name:
                    init.zeros_(param)

    def update_A(self):
        S = self.structure_learner.generate_A(self.data.x, self.data.edge_index)  # range is [-1, 1]
        A = self.tau * self.A + (1 - self.tau) * S
        # A = torch.sigmoid(A)
        A = (A > self.threshold).float()
        self.A = A.to(self.device)
        
    def train_epoch(self, epoch):
        if epoch % 2 == 0:
            self.update_A()
        x_edge_index = torch.nonzero(self.A).t()
        x = self.encoder(self.data.x, x_edge_index)
        x_aug_edge_index = self.structure_learner(self.data.x, self.data.edge_index)
        x_aug = self.encoder(self.data.x, x_aug_edge_index)
        loss = self.calc_loss(x, x_aug, temperature=0.3)
        return loss

    def calc_loss(self, x, x_aug, temperature):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        epsilon = 1e-8
        if x_abs.min() == 0.0:
            x_abs = x_abs.clamp(min=epsilon)
        if x_aug_abs.min() == 0.0:
            x_aug_abs = x_aug_abs.clamp(min=epsilon)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss_0 = - torch.log(loss_0).mean()
        loss_1 = - torch.log(loss_1).mean()
        loss = (loss_0 + loss_1) / 2.0
        return loss
    

















