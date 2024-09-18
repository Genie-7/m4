import os
import torch
import dgl

def check_fashionmnist_data(data_dir):
    splits = ['train', 'val', 'test']
    
    for split in splits:
        file_path = os.path.join(data_dir, f"{split}.pt")
        
        if not os.path.exists(file_path):
            print(f"Error: {split}.pt file not found in {data_dir}")
            continue
        
        print(f"Checking {split}.pt...")
        data = torch.load(file_path)
        
        if 'graphs' not in data or 'labels' not in data:
            print(f"Error: {split}.pt does not contain 'graphs' and 'labels' keys")
            continue
        
        graphs = data['graphs']
        labels = data['labels']
        
        print(f"Number of graphs in {split} set: {len(graphs)}")
        print(f"Number of labels in {split} set: {len(labels)}")
        
        sample_graph = graphs[0]
        
        if not isinstance(sample_graph, dgl.DGLGraph):
            print(f"Error: Graphs in {split}.pt are not dgl.DGLGraph objects")
            continue
        
        print(f"Attributes of the first graph in {split} set:")
        for attr in sample_graph.ndata.keys():
            print(f"  {attr}: {sample_graph.ndata[attr].shape}")
        
        if 'pos' in sample_graph.ndata:
            print("  'pos' attribute is present")
        else:
            print("  Warning: 'pos' attribute is missing")
        
        print(f"Number of nodes: {sample_graph.num_nodes()}")
        print(f"Number of edges: {sample_graph.num_edges()}")
        if 'feat' in sample_graph.ndata:
            print(f"Node features shape: {sample_graph.ndata['feat'].shape}")
        print(f"Label: {labels[0]}")
        print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check FashionMNIST dataset structure")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the .pt files')
    args = parser.parse_args()
    
    check_fashionmnist_data(args.data_dir)