import argparse
def main(args):
    if args.dataset_name == "Cora":
        # import sys
        # import os
        # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
        from torch_geometric.datasets import Planetoid
        path = args.data_path  # your path
        dataset = Planetoid(root=path, name='Cora')
        data = dataset[0]
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Feature Dimension: {data.num_node_features}")
        print(f"Number of classes: {dataset.num_classes}")
    elif args.dataset_name == "Citeseer":
        # import sys
        # import os
        # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
        from torch_geometric.datasets import Planetoid
        path = args.data_path  # your path
        dataset = Planetoid(root=path, name='Citeseer')
        data = dataset[0]
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Feature Dimension: {data.num_node_features}")
        print(f"Number of classes: {dataset.num_classes}")
    elif args.dataset_name == "PubMed":
        # import sys
        # import os
        # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
        from torch_geometric.datasets import Planetoid
        path = args.data_path  # your path
        dataset = Planetoid(root=path, name='PubMed')
        data = dataset[0]
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Feature Dimension: {data.num_node_features}")
        print(f"Number of classes: {dataset.num_classes}")
    elif args.dataset_name == "Texas":
        from torch_geometric.datasets import WebKB
        path = args.data_path
        dataset = WebKB(root=path, name='Texas')
        data = dataset[0]
    elif args.dataset_name == "Cornell":
        from torch_geometric.datasets import WebKB
        path = args.data_path
        dataset = WebKB(root=path, name='Cornell')
        data = dataset[0]
    elif args.dataset_name == "Wisconsin":
        from torch_geometric.datasets import WebKB
        path = args.data_path
        dataset = WebKB(root=path, name='Wisconsin')
        data = dataset[0]
    elif args.dataset_name == "Actor":
        from torch_geometric.datasets import Actor
        path = args.data_path
        dataset = Actor(root=path)
        data = dataset[0]
    elif args.dataset_name == "Chameleon":
        from torch_geometric.datasets import WikipediaNetwork
        path = args.data_path
        dataset = WikipediaNetwork(root=path, name='chameleon')
        data = dataset[0]
    elif args.dataset_name == "Squirrel":
        from torch_geometric.datasets import WikipediaNetwork
        path = args.data_path
        dataset = WikipediaNetwork(root=path, name='squirrel')
        data = dataset[0]
    elif args.dataset_name == "ENZYMES":
        from torch_geometric.datasets import TUDataset
        path = args.data_path
        dataset = TUDataset(root=path, name='ENZYMES')
        data = dataset[0]
    elif args.dataset_name == "PROTEINS":
        from torch_geometric.datasets import TUDataset
        path = args.data_path
        dataset = TUDataset(root=path, name='PROTEINS')
        data = dataset[0]
    elif args.dataset_name == "MUTAG":
        from torch_geometric.datasets import TUDataset
        path = args.data_path
        dataset = TUDataset(root=path, name='MUTAG')
        data = dataset[0]
    elif args.dataset_name == "COX2":
        from torch_geometric.datasets import TUDataset
        path = args.data_path
        dataset = TUDataset(root=path, name='COX2')
        data = dataset[0]
    elif args.dataset_name == "BZR":
        from torch_geometric.datasets import TUDataset
        path = args.data_path
        dataset = TUDataset(root=path, name='BZR')
        data = dataset[0]
    else:
        raise NotImplementedError(f"Dataset {args.dataset_name} is not implemented.")

if __name__ == "__main__":
    # Download the original dataset.
    parser = argparse.ArgumentParser()
    # Node_level: [Cora, Citeseer, PubMed, Texas, Cornell, Wisconsin, Actor, Chameleon, Squirrel]
    # Graph_level: [ENZYMES, PROTEINS, MUTAG, COX2, BZR]
    parser.add_argument('--dataset_name', dest='dataset_name', type=str, help='name of dataset', default="Cora")
    parser.add_argument('--data_path', dest='data_path', type=str, help='path of original data', default="./Original_data")
    args = parser.parse_args()
    main(args)
    

