import torch
import numpy as np
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected

class RandomWalkSubgraph:
    def __init__(self, num_walks=10, walk_length=5):
        self.num_walks = num_walks
        self.walk_length = walk_length
    
    def get_khop_neighbors(self, edge_index, node_idx, k, num_nodes):
        """Get k-hop neighbors of given nodes"""
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes))
        
        # Initialize masks
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
        mask[node_idx] = True
        
        # Get k-hop neighbors
        for _ in range(k):
            mask = (adj.matmul(mask.float()) > 0)
        return torch.nonzero(mask).squeeze()

    def weighted_random_walk(self, edge_index, start_nodes, weights, num_nodes):
        """Perform weighted random walks starting from seed nodes"""
        device = edge_index.device
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes))
        
        all_walks = []
        for _ in range(self.num_walks):
            current_nodes = start_nodes
            walk = [current_nodes]
            
            for _ in range(self.walk_length):
                # Get neighbors of current nodes
                row, col, _ = adj.coo()
                mask = torch.isin(row, current_nodes)
                neighbors = col[mask]
                
                if len(neighbors) == 0:
                    break
                    
                # Apply weights to neighbor selection
                neighbor_weights = weights[neighbors]
                neighbor_weights = torch.softmax(neighbor_weights, dim=0)
                
                # Sample next nodes based on weights
                probs = neighbor_weights.cpu().numpy()
                next_nodes = np.random.choice(
                    neighbors.cpu().numpy(), 
                    size=len(current_nodes),
                    p=probs if len(probs) > 0 else None
                )
                current_nodes = torch.tensor(next_nodes, device=device)
                walk.append(current_nodes)
            
            all_walks.append(torch.cat(walk))
            
        return torch.cat(all_walks).unique()

    def get_subgraph(self, edge_index, node_pairs, k1_weight, k2_weight, num_nodes):
        """Generate subgraph using weighted k-hop neighbors as random walk seeds"""
        device = edge_index.device
        subgraph_nodes = []
        
        for src, dst in node_pairs.t():
            # Get k1 and k2-hop neighbors
            k1_nodes = self.get_khop_neighbors(edge_index, [src, dst], 1, num_nodes)
            k2_nodes = self.get_khop_neighbors(edge_index, [src, dst], 2, num_nodes)
            
            # Create weighted seed nodes
            seed_nodes = torch.cat([k1_nodes, k2_nodes])
            weights = torch.ones(len(seed_nodes), device=device)
            weights[:len(k1_nodes)] *= k1_weight
            weights[len(k1_nodes):] *= k2_weight
            
            # Perform weighted random walks
            subgraph = self.weighted_random_walk(edge_index, seed_nodes, weights, num_nodes)
            subgraph_nodes.append(subgraph)
            
        # Combine all subgraph nodes
        all_nodes = torch.cat(subgraph_nodes).unique()
        
        # Extract subgraph edges
        row, col = edge_index
        mask = torch.isin(row, all_nodes) & torch.isin(col, all_nodes)
        subgraph_edge_index = edge_index[:, mask]
        
        return all_nodes, subgraph_edge_index 