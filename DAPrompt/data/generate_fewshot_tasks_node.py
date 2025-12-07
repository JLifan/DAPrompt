from collections import defaultdict
import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data, Dataset
import sys
import os
from torch_geometric.datasets import Planetoid
import argparse
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import warnings
warnings.filterwarnings("ignore")
from prompt_learning.ours.DAPrompt.utils import set_seed, sample_Nway_Kshot_task
from tqdm import tqdm
import time


class SubgraphDataset(Dataset):
    def __init__(self, full_data, center_nodes, labels=None, num_hops=3):
        super().__init__(root=".")
        self.full_data = full_data
        self.center_nodes = center_nodes.tolist() if isinstance(center_nodes, torch.Tensor) else center_nodes
        self.labels = labels
        self.num_hops = num_hops

    def __len__(self):
        return len(self.center_nodes)

    def __getitem__(self, idx):
        center = self.center_nodes[idx]
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
            center, self.num_hops, self.full_data.edge_index, relabel_nodes=True
        )
        x = self.full_data.x[sub_nodes]
        y = self.labels[idx] if self.labels is not None else -1
        return Data(x=x, edge_index=sub_edge_index, y=y, center_id=mapping)


class TrainDataset(Dataset):
    def __init__(self, support_dataset):
        super().__init__(root='.')
        self.support = support_dataset

    def __len__(self):
        return len(self.support)

    def __getitem__(self, idx):
        return self.support[idx]


class TestDataset(Dataset):
    def __init__(self, query_dataset):
        super().__init__(root='.')
        self.query = query_dataset

    def __len__(self):
        return len(self.query)
    
    def __getitem__(self, idx):
        return self.query[idx]


class ValDataset(Dataset):
    def __init__(self, val_dataset):
        super().__init__(root='.')
        self.val = val_dataset

    def __len__(self):
        return len(self.val)
    
    def __getitem__(self, idx):
        return self.val[idx]


class TaskDataset:
    def __init__(self, dataset, num_tasks=50):
        self.dataset = dataset
        self.num_tasks = num_tasks

    def __iter__(self):
        for i in range(self.num_tasks):
            train_data = TrainDataset(self.dataset[i][0])
            test_data = TestDataset(self.dataset[i][1])
            yield train_data, test_data


class RepetitionDataset:
    def __init__(self, data_path, num_repetitions):
        self.data_path = data_path
        self.num_repetitions = num_repetitions
        self.data = []
        self.init()

    def init(self):
        print("[INFO] Starting to load all rounds and repetitions data...")
        self.data = torch.load(self.data_path)

    def __iter__(self):
        for i in range(self.num_repetitions):
            yield TaskDataset(self.data[i])


def create_subgraph(data, indices, lables, hop):
    data_list = []
    for _, (index, lable) in enumerate(zip(indices, lables)):
        if index.dim() == 0:
            index = index.unsqueeze(0)
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(index, hop, data.edge_index, relabel_nodes=True)
        x = data.x[sub_nodes]
        y = torch.tensor(lable)
        data_list.append(Data(x=x, edge_index=sub_edge_index, y=y, center_id=mapping))
    return data_list


def generate_tasks(args):
    if args.dataset_name in ["Cora", "CiteSeer", "PubMed"]:
        if args.data_path == "./":
            raise FileNotFoundError(f"Path does not exist: {args.data_path}")
        original_data = Planetoid(
            root=args.data_path,
            name=args.dataset_name
        )
        original_data = original_data[0]
    elif args.dataset_name == "Squirrel":
        if args.data_path == "./":
            raise FileNotFoundError(f"Path does not exist: {args.data_path}")
        from torch_geometric.datasets import WikipediaNetwork
        original_data = WikipediaNetwork(root=args.data_path, name='squirrel')[0]
    elif args.dataset_name == "Chameleon":
        if args.data_path == "./":
            raise FileNotFoundError(f"Path does not exist: {args.data_path}")
        from torch_geometric.datasets import WikipediaNetwork
        original_data = WikipediaNetwork(root=args.data_path, name='chameleon')[0]
    elif args.dataset_name == "Actor":
        if args.data_path == "./":
            raise FileNotFoundError(f"Path does not exist: {args.data_path}")
        from torch_geometric.datasets import Actor
        original_data = Actor(root=args.data_path)[0]
    elif args.dataset_name == "Cornell":
        if args.data_path == "./":
            raise FileNotFoundError(f"Path does not exist: {args.data_path}")
        from torch_geometric.datasets import WebKB
        original_data = WebKB(root=args.data_path, name='Cornell')[0]
    elif args.dataset_name == "Texas":
        args.N = 4
        if args.data_path == "./":
            raise FileNotFoundError(f"Path does not exist: {args.data_path}")
        from torch_geometric.datasets import WebKB
        original_data = WebKB(root=args.data_path, name='Texas')[0]
    elif args.dataset_name == "Wisconsin":
        args.N = 4
        if args.data_path == "./":
            raise FileNotFoundError(f"Path does not exist: {args.data_path}")
        from torch_geometric.datasets import WebKB
        original_data = WebKB(root=args.data_path, name='Wisconsin')[0]
    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset_name}")

    label_to_indices = defaultdict(list)
    for idx, label in enumerate(original_data.y.tolist()):
        if args.dataset_name == "Wisconsin":
            if label == 0:  # In the Wisconsin dataset, label 0 is invalid; this class has only 10 nodes.
                continue
            elif label == 4:
                label_to_indices[0].append(idx)  # Labels must be consecutive!
            else:
                label_to_indices[label].append(idx)
        elif args.dataset_name == "Texas":
            if label == 1:  # In the Texas dataset, label 1 is invalid; this class has only one node.
                continue
            elif label == 4:
                label_to_indices[1].append(idx)  # Labels must be consecutive!
            else:
                label_to_indices[label].append(idx)
        else:
            label_to_indices[label].append(idx)
    print(f"Data loaded with {original_data.num_nodes} nodes and {original_data.num_edges} edges.")
    print(label_to_indices.keys())
    for key, value in label_to_indices.items():
        print(f"Number of nodes in label {key}: {len(value)}")
    repetition = []
    for repeat in tqdm(range(args.repetition), desc="Repetition number", leave=False):
        tasks = []
        for i in tqdm(range(args.task), desc="Tasks number", leave=False):  # 50 N-way K-shot tasks
            support_idx, query_idx, support_labels ,query_labels = sample_Nway_Kshot_task(label_to_indices=label_to_indices, way=args.N, shot=args.K, num_query_per_class=args.Q)
            train_data = create_subgraph(original_data, support_idx, support_labels, hop=args.hop)
            test_data = create_subgraph(original_data, query_idx, query_labels, hop=args.hop)
            task = [train_data, test_data]
            tasks.append(task)
        repetition.append(tasks)
    torch.save(repetition, args.save_path)
    print(f"Tasks generated and saved to {args.save_path}")
    return

if __name__ == "__main__":
    set_seed(2026)
    
    # Node_level: [Cora:7, Citeseer:6, PubMed:3, Texas:5, Cornell:5, Wisconsin:5, Actor:5, Chameleon:5, Squirrel:5]
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', dest='dataset_name', type=str, help='name of dataset', default="Squirrel")
    parser.add_argument('--data_path', dest='data_path', type=str, help='path of original data', default="./")
    parser.add_argument('--N', dest='N', type=int, help='N-way', default=5)  # Equivalent to how many classes the dataset has
    parser.add_argument('--K', dest='K', type=int, help='K-shot', default=1)  # 1 or 5
    parser.add_argument('--Q', dest='Q', type=int, help='Q-query', default=10)
    parser.add_argument('--seed', dest='seed', type=int, help='seed number', default=2026)
    parser.add_argument('--repetition', dest='repetition', type=int, help='number of repetition to calculate mean and std (default is 10)', default=10)
    parser.add_argument('--task', dest='task', type=int, help='number of tasks included in each repetition (default is 50)', default=50)
    parser.add_argument('--hop', dest='hop', type=int, help='K-hop of subgraph', default=2)
    args = parser.parse_args()

    args.save_path = (
        "./"  # you need to change this path to your desired save location
        "Node_level/" + args.dataset_name + f"/{args.dataset_name}_{args.K}shot.pt"
        )
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    start_time = time.time()
    generate_tasks(args)
    end_time = time.time()
    print(f"Tasks generated in {end_time - start_time:.2f} seconds.")