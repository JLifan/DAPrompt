import random
import torch
import numpy as np
from torch_geometric.utils import add_self_loops, to_dense_adj

def set_seed(seed: int = 2026):
    random.seed(seed)               
    np.random.seed(seed)           
    torch.manual_seed(seed)             
    torch.cuda.manual_seed(seed)       
    torch.cuda.manual_seed_all(seed)    
    
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  


def sample_Nway_Kshot_task(label_to_indices, way, shot, num_query_per_class=10):

    classes = random.sample(list(label_to_indices.keys()), way)
    support_idx, query_idx, support_labels ,query_labels = [], [], [], []

    for new_label, cls in enumerate(classes):
        candidates = random.sample(label_to_indices[cls], shot + num_query_per_class)
        support_idx += candidates[:shot]
        support_labels += [new_label] * shot
        query_idx += candidates[shot:]
        query_labels += [new_label] * num_query_per_class

    indices_support = list(range(len(support_idx)))
    indices_query = list(range(len(query_idx)))
    random.shuffle(indices_support)
    random.shuffle(indices_query)
    support_idx = [support_idx[i] for i in indices_support]
    query_idx = [query_idx[i] for i in indices_query]
    support_labels = [support_labels[i] for i in indices_support]
    query_labels = [query_labels[i] for i in indices_query]

    support_idx = torch.tensor(support_idx)
    query_idx = torch.tensor(query_idx)
    support_labels = torch.tensor(support_labels)
    query_labels = torch.tensor(query_labels)

    return support_idx, query_idx, support_labels, query_labels



def prune_graph(data, num_nodes_to_remove):
    """
    Randomly removes a subset of nodes and their associated edges from a graph.
    :param data: A PyG data object containing node features, edge indices, etc.
    :param num_nodes_to_remove: The number of nodes to remove.
    :return: A new graph data object after node removal.
    """

    num_nodes = data.x.size(0)


    nodes_to_remove = random.sample(range(num_nodes), num_nodes_to_remove)
    nodes_to_remove = set(nodes_to_remove) 

    edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    data.edge_index = edge_index

    edge_index = data.edge_index

    mask = ~((edge_index[0].unsqueeze(1) == torch.tensor(list(nodes_to_remove)).view(1, -1)).any(dim=1) |
             (edge_index[1].unsqueeze(1) == torch.tensor(list(nodes_to_remove)).view(1, -1)).any(dim=1))

    new_edge_index = edge_index[:, mask]

    remaining_nodes = list(set(range(num_nodes)) - nodes_to_remove)

    new_x = data.x[remaining_nodes]
    new_y = data.y[remaining_nodes]

    new_edge_index = new_edge_index.clone()
    for i in range(new_edge_index.size(1)):
        new_edge_index[0, i] = remaining_nodes.index(new_edge_index[0, i].item())
        new_edge_index[1, i] = remaining_nodes.index(new_edge_index[1, i].item())

    new_data = data.__class__(
        x=new_x,
        edge_index=new_edge_index,
        y=new_y
    )

    return new_data












