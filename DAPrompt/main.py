from torch_geometric.data import DataLoader, Batch
from torch_geometric.datasets import Planetoid
from collections import defaultdict
import random
import torch
import numpy as np
import torch.nn.functional as F
import sys
import os
import argparse
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import warnings
warnings.filterwarnings("ignore")
import copy
from prompt_learning.ours.DAPrompt.data.generate_fewshot_tasks_node import RepetitionDataset
from utils import set_seed
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from backbone import GCNEncoder, GATEncoder, GraphSAGEEncoder, GraphTransformerEncoder, GeomGCNEncoder
from pretraining import Structure_pretrain
from prompt import DAPrompt


def run_single_task(train_data, test_data, args, writer):

    # choose the backbone model, GCN is the default
    if args.model_name == "GCN":
        model = GCNEncoder(
            in_channels=args.in_dim, 
            hidden_channels=args.hidden_dim, 
            out_channels=args.N
            ).to(args.device)
    elif args.model_name == "GAT":
        model = GATEncoder(
            in_dim=args.in_dim, 
            hidden_dim=args.hidden_dim, 
            out_dim=args.N
            ).to(args.device)
    elif args.model_name == "GeomGCN":      
        model = GeomGCNEncoder(
            in_dim=args.in_dim, 
            hidden_dim=args.hidden_dim, 
            out_dim=args.N
            ).to(args.device)
    elif args.model_name == "GraphSAGE":
        model = GraphSAGEEncoder(
            in_dim=args.in_dim, 
            hidden_dim=args.hidden_dim, 
            out_dim=args.N
            ).to(args.device)
    elif args.model_name == "GraphTransformer":
        model = GraphTransformerEncoder(
            in_dim=args.in_dim, 
            hidden_dim=args.hidden_dim, 
            out_dim=args.N
            ).to(args.device)
    else:
        raise ValueError(f"Unknown model: {args.model_name}")
    
    # load pretrained model
    if not os.path.isfile(args.pre_save_path):
        print("[INFO] Pretrained model not found. Starting pretraining...")
        # pretrain model
        if args.pre_backbone_name == "GCN":
            backbone = GCNEncoder(
                in_channels=args.in_dim, 
                hidden_channels=args.hidden_dim, 
                out_channels=args.N
                ).to(args.device)
        elif args.pre_backbone_name == "GAT":
            backbone = GATEncoder(
                in_dim=args.in_dim, 
                hidden_dim=args.hidden_dim, 
                out_dim=args.N
                ).to(args.device)
        elif args.pre_backbone_name == "GeomGCN":
            backbone = GeomGCNEncoder(
                in_dim=args.in_dim, 
                hidden_dim=args.hidden_dim, 
                out_dim=args.N
                ).to(args.device)
        elif args.pre_backbone_name == "GraphSAGE":
            backbone = GraphSAGEEncoder(
                in_dim=args.in_dim, 
                hidden_dim=args.hidden_dim, 
                out_dim=args.N
                ).to(args.device)
        elif args.pre_backbone_name == "GraphTransformer":
            backbone = GraphTransformerEncoder(
                in_dim=args.in_dim, 
                hidden_dim=args.hidden_dim, 
                out_dim=args.N
                ).to(args.device)
        else:
            raise ValueError(f"Unknown backbone model: {args.pre_backbone_name}")
        
        if args.pre_Method == "DAPrompt":
            method = Structure_pretrain(
                backbone, 
                args.data.x.shape[0], 
                args.pre_tau, 
                args.data, 
                args.pre_threshold, 
                args.device
                ).to(args.device)
        else:
            raise ValueError(f"Unknown pretrain method: {args.pre_Method}")
        
        # record of pretrain
        log_dir_pre = args.pre_record_path
        if not os.path.exists(log_dir_pre):
            os.makedirs(log_dir_pre)
        print(f"Logging to {log_dir_pre}")
        writer_pre = SummaryWriter(log_dir=log_dir_pre)
        pretrain_model(method=method, save_path=args.pre_save_path, args=args, writer=writer_pre)
    
    model.load_state_dict(torch.load(args.pre_save_path, map_location=args.device))

    daprompt_model = DAPrompt(
                              encoder=model,
                              structure_token_number=args.structure_token_number, 
                              semantics_token_number=args.N,
                              input_dim=args.in_dim, 
                              hid_dim=args.hidden_dim, 
                              device=args.device, 
                              outer_thre=args.outer_thre,
                              lamda1=args.lambda1,
                              lamda2=args.lambda2,
                              structure_threshold=args.structure_thre,
                              broadcast=args.broadcast,
                              ).to(args.device)
    optimizer = torch.optim.Adam(daprompt_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dataset = train_data
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    for epoch in tqdm(range(args.Epoch), desc="Prompts Tuning Epochs", leave=False):
        daprompt_model.train()
        losses = 0.0
        for batch in dataloader:
            batch = batch.to(args.device)
            loss = daprompt_model.loss(batch)
            losses += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        writer.add_scalar("Loss/train", losses, epoch + 1)  

    test_dataset = test_data
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(args.device)
            logits = daprompt_model(batch)
            pred = logits.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

    return correct / total


def pretrain_model(method, save_path, args, writer):
    optimizer = torch.optim.Adam(method.parameters(), lr=args.pre_lr, weight_decay=args.pre_weight_decay)
    method.train()
    method.to(args.device)
    print("Start pretraining...")

    best_loss = float('inf')
    best_state_dict = None
    patience_counter = 0
    patience = args.Patience if hasattr(args, "Patience") else args.patience

    if args.pre_Method == "DAPrompt":
        for epoch in range(args.pre_Epoch):
            optimizer.zero_grad()
            loss = method.train_epoch(epoch)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}/{args.pre_Epoch}, Loss: {loss.item():.4f}")
            writer.add_scalar('Loss/train', loss.item(), epoch)

            if loss.item() < best_loss - 1e-4:
                best_loss = loss.item()
                best_state_dict = method.encoder.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}, best loss = {best_loss:.4f}")
                    break

        print("Pretraining finished.")

    # save the best model
    if best_state_dict is not None:
        torch.save(best_state_dict, save_path)
    else:
        torch.save(method.encoder.state_dict(), save_path)
    return


def process_args(args):

    if args.dataset_name == "Cora":
        args.in_dim = 1433
    elif args.dataset_name == "CiteSeer":
        args.in_dim = 3703
    elif args.dataset_name == "PubMed": 
        args.in_dim = 500
    elif args.dataset_name == "Wisconsin":
        args.in_dim = 1703
        args.N = 4
    elif args.dataset_name == "Texas":
        args.in_dim = 1703
        args.N = 4
    elif args.dataset_name == "Cornell":
        args.in_dim = 1703
    elif args.dataset_name == "Chameleon":
        args.in_dim = 2325
    elif args.dataset_name == "Squirrel":
        args.in_dim = 2089
    elif args.dataset_name == "Actor":
        args.in_dim = 932
    elif args.dataset_name == 'ENZYMES':
        args.in_dim = 3
    elif args.dataset_name == 'PROTEINS':
        args.in_dim = 3
    elif args.dataset_name == 'MUTAG':
        args.in_dim = 7
    elif args.dataset_name == 'COX2':
        args.in_dim = 35
    elif args.dataset_name == 'BZR':
        args.in_dim = 53
    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset_name}")

    args.edge_relation = "relation"

    if not os.path.isfile(args.pre_save_path):
        if args.dataset_name in ["Cora", "CiteSeer", "PubMed"]:
            if args.pre_data_path == "./":
                # args.pre_data_path = "/home/main/dataset_temp/Original_data"  # Directory containing the original dataset
                raise FileNotFoundError(f"Path does not exist: {args.pre_data_path}")
            original_data = Planetoid(
                root=args.pre_data_path,
                name=args.dataset_name
            )
            original_data = original_data[0]
        elif args.dataset_name == "Squirrel":
            if args.pre_data_path == "./":
                raise FileNotFoundError(f"Path does not exist: {args.pre_data_path}")
            from torch_geometric.datasets import WikipediaNetwork
            original_data = WikipediaNetwork(root=args.pre_data_path, name='squirrel')[0]
        elif args.dataset_name == "Chameleon":
            if args.pre_data_path == "./":
                raise FileNotFoundError(f"Path does not exist: {args.pre_data_path}")
            from torch_geometric.datasets import WikipediaNetwork
            original_data = WikipediaNetwork(root=args.pre_data_path, name='chameleon')[0]
        elif args.dataset_name == "Actor":
            if args.pre_data_path == "./":
                raise FileNotFoundError(f"Path does not exist: {args.pre_data_path}")
            from torch_geometric.datasets import Actor
            original_data = Actor(root=args.pre_data_path)[0]
        elif args.dataset_name == "Cornell":
            if args.pre_data_path == "./":
                raise FileNotFoundError(f"Path does not exist: {args.pre_data_path}")
            from torch_geometric.datasets import WebKB
            original_data = WebKB(root=args.pre_data_path, name='Cornell')[0]
        elif args.dataset_name == "Texas":
            args.N = 4
            if args.pre_data_path == "./":
                raise FileNotFoundError(f"Path does not exist: {args.pre_data_path}")
            from torch_geometric.datasets import WebKB
            original_data = WebKB(root=args.pre_data_path, name='Texas')[0]
        elif args.dataset_name == "Wisconsin":
            args.N = 4
            if args.pre_data_path == "./":
                raise FileNotFoundError(f"Path does not exist: {args.pre_data_path}")
            from torch_geometric.datasets import WebKB
            original_data = WebKB(root=args.pre_data_path, name='Wisconsin')[0]
        elif args.dataset_name in ['ENZYMES', 'PROTEINS', 'MUTAG', 'COX2', 'BZR']:
            if args.pre_data_path == "./":
                raise FileNotFoundError(f"Path does not exist: {args.pre_data_path}")
            data_path = args.pre_data_path
            from torch_geometric.datasets import TUDataset
            dataset = TUDataset(root=data_path, name=args.dataset_name)
            some_dataset = dataset
            # some_dataset = dataset[:some_len]  # if you out of memory, you can use this to reduce the data size 

            # indices = torch.randperm(len(dataset))[:some_len]
            # some_dataset = [dataset[i] for i in indices]

            original_data = Batch.from_data_list(list(some_dataset))
        else:
            raise ValueError(f"Unsupported dataset name: {args.dataset_name}")
        
        args.data = original_data
    else:
        args.data = None

    return args


def main(args):

    if not torch.cuda.is_available():
        args.device = torch.device("cpu")
    print(f"Using device: {args.device}")

    if args.pre_record_path == "./":
        args.pre_record_path = f"./runs_pretrain/{args.pre_Method}_{args.pre_backbone_name}_{args.dataset_name}"
    if args.record_path == "./":
        args.record_path = f"./runs_downstream/{args.pre_Method}_{args.model_name}_{args.dataset_name}_N{args.N}_K{args.K}_Q{args.Q}_hop{args.hop}"
    # record of downstream
    log_dir_down = args.record_path
    if not os.path.exists(log_dir_down):
        os.makedirs(log_dir_down)
    print(f"Logging to {log_dir_down}")
    writer_down = SummaryWriter(log_dir=log_dir_down)

    # load dataset
    repetitionDataset = RepetitionDataset(args.data_path, args.repetition)
    print("Dataset has been successfully loaded.")

    # time
    starttime = time.time()

    all_accuracies = []

    for repeat, taskDataset in enumerate(repetitionDataset):
        accs = []
        for i, (train_Data, test_data) in enumerate(taskDataset): 
            
            acc = run_single_task(train_Data, test_data, args, writer_down)
            
            writer_down.add_scalar("Task/accuracy", acc, i + 1)

            print(f"Repeat {repeat+1}/Task {i + 1}: Accuracy = {acc:.4f}")
            accs.append(acc)
        mean_acc = np.mean(accs)
        all_accuracies.append(mean_acc)
        print(f"Repeat {repeat + 1}: Mean Accuracy over 50 tasks = {mean_acc:.4f}")
    
    endtime = time.time()
    args.label_to_indices = None
    args.optimizer = None
    print(args)

    overall_mean = np.mean(all_accuracies)
    overall_std = np.std(all_accuracies)
    print(f"\nFinal Result: Accuracy = {overall_mean:.4f} ± {overall_std:.4f}")
    writer_down.add_scalar("Accuracy/final_mean", np.mean(all_accuracies), 0)
    writer_down.add_scalar("Accuracy/final_std", np.std(all_accuracies), 0)
    writer_down.add_scalar("Time/total", endtime - starttime, 0)
    writer_down.close()
    
    time_diff = endtime - starttime

    hours = int(time_diff // 3600)
    minutes = int((time_diff % 3600) // 60)
    seconds = int(time_diff % 60)

    print(f"Total time: {hours} hours {minutes} minutes {seconds} seconds")

    
 



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Node_level: [Cora:7, Citeseer:6, PubMed:3, Texas:5, Cornell:5, Wisconsin:5, Actor:5, Chameleon:5, Squirrel:5]
    # Graph_level: [ENZYMES:6, PROTEINS:2, MUTAG:2, COX2:2, BZR:2]
    parser.add_argument('--dataset_name', dest='dataset_name', type=str, help='name of dataset', default="Cornell")
    parser.add_argument('--data_path', dest='data_path', type=str, help='path of downstream dataset', default="./data/node_level/Cornell_5shot.pt")
    # model [GCN, GAT, GeomGCN, GraphSAGE, GraphTransformer]
    parser.add_argument('--model_name', dest='model_name', type=str, help='name of model', default="GCN")
    parser.add_argument('--N', dest='N', type=int, help='N-way', default=5)
    parser.add_argument('--K', dest='K', type=int, help='K-shot', default=5)
    parser.add_argument('--Q', dest='Q', type=int, help='Q-query', default=10)
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='defualt is 2(5shot) and 64(100shot)', default=4)
    parser.add_argument('--seed', dest='seed', type=int, help='seed number', default=2026)
    parser.add_argument('--repetition', dest='repetition', type=int, help='number of repetition to calculate mean and std, default is 10', default=10)
    parser.add_argument('--task', dest='task', type=int, help='every repetition include how many tasks, default is 100', default=50)
    parser.add_argument('--Epoch', dest='Epoch', type=int, help='every task train how many epochs', default=20)
    parser.add_argument('--hop', dest='hop', type=int, help='K-hop of subgraph', default=2)
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, help='hidden_dim', default=32)
    parser.add_argument('--structure_token_number', dest='structure_token_number', type=int, help='structure_token_number', default=2)
    parser.add_argument('--outer_thre', dest='outer_thre', type=float, help='structure rewire threshold', default=0.2)
    parser.add_argument('--lambda1', dest='lambda1', type=float, help='loss weights', default=0.5)
    parser.add_argument('--lambda2', dest='lambda2', type=float, help='loss weights', default=0.5)
    parser.add_argument('--structure_thre', dest='structure_thre', type=float, help='structure threshold', default=0.8)
    parser.add_argument('--lr', dest='lr', type=float, help='learning rate', default=0.01)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, help='weight decay', default=5e-4)
    parser.add_argument('--record_path', dest='record_path', type=str, help='where to save logs for downstream run', default="./")

    parser.add_argument('--pre_Method', dest='pre_Method', type=str, help='pretrain paradigm', default="DAPrompt")
    parser.add_argument('--pre_save_path', dest='pre_save_path', type=str, help='where to save pretrained the model', default="./weights/DAPrompt_GCN_Cornell_K5_best.pt")
    parser.add_argument('--pre_record_path', dest='pre_record_path', type=str, help='where to save logs for pretraining run', default="./")
    parser.add_argument('--pre_backbone_name', dest='pre_backbone_name', type=str, help='name of backbone model', default="GCN")
    parser.add_argument('--pre_data_path', dest='pre_data_path', type=str, help='path of pretrain dataset', default="./")
    parser.add_argument('--pre_Epoch', dest='pre_Epoch', type=int, help='how many epochs', default=1000)
    parser.add_argument('--pre_tau', dest='pre_tau', type=float, help='pretrain update weight', default=0.2)
    parser.add_argument('--pre_threshold', dest='pre_threshold', type=float, help='pretrain threshold', default=0.1)
    parser.add_argument('--pre_lr', dest='pre_lr', type=float, help='learning rate', default=0.0001)
    parser.add_argument('--pre_weight_decay', dest='pre_weight_decay', type=float, help='weight decay', default=5e-4)
    parser.add_argument('--pre_patience', dest='pre_patience', type=int, help='early stopping', default=50)
    parser.add_argument('--broadcast', dest='broadcast', type=bool, help='broadcast semantics', default=True)
    parser.add_argument('--device', dest='device', type=str, help='GPU or CPU or ...', default="cuda:0")
    parser.add_argument('--shuffle', dest='shuffle', type=bool, help='shuffle dataloader', default=True)
    

    args = parser.parse_args()


    args = process_args(args)
    set_seed(args.seed)
    main(args)
    


















































