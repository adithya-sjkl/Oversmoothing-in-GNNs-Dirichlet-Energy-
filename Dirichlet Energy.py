import torch
import torch_geometric
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import to_dense_adj

def change_nonzero_to_one(tensor):
    # Create a mask for nonzero elements
    mask = tensor != 0
    # Change nonzero elements to 1 using the mask
    tensor[mask] = 1
    return tensor


def dirichlet_energy(activations, edge_index, num_nodes):
    """
    Calculates the Dirichlet energy of the hidden layer activations in a GNN.

    Args:
        activations (torch.Tensor): A matrix of size (num_nodes, hidden_dim) containing the hidden layer activations.
        edge_index (torch.Tensor): The edge index of the graph.
        num_nodes (int): The number of nodes in the graph.

    Returns:
        torch.Tensor: The Dirichlet energy of the hidden layer activations.
    """
    edge_index_sl,_  = add_self_loops(edge_index, num_nodes=num_nodes)
    adj = to_dense_adj(edge_index_sl , max_num_nodes=num_nodes)
    adj = change_nonzero_to_one(adj) # sometimes we get values other than 1 due to duplication
    adj = adj.squeeze()
    degree_matrix_root = torch.diag(torch.sqrt(adj.sum(dim=1)).squeeze())
    deg = torch.linalg.inv(degree_matrix_root)

    laplacian = torch.eye(num_nodes) - deg @ adj @ deg

    dirichlet_energy = torch.trace( activations.t() @ laplacian @ activations )


    return dirichlet_energy
