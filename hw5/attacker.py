import torch
import numpy as np
import scipy.sparse as sp
from torch.nn.modules.module import Module


class BaseAttack(Module):
    """Abstract base class for target attack classes.
    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    device: str
        'cpu' or 'cuda'
    """

    def __init__(self, model, nnodes, device='cpu'):
        super(BaseAttack, self).__init__()
        self.surrogate = model
        self.nnodes = nnodes
        self.device = device

        self.modified_adj = None

    def attack(self, ori_adj, n_perturbations, **kwargs):
        """Generate perturbations on the input graph.
        Parameters
        ----------
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix.
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        Returns
        -------
        None.
        """
        raise NotImplementedError()

# TODO: Implemnet your own attacker here
# class MyAttacker(BaseAttack):
#     def __init__(self, model=None, nnodes=None, device='cpu'):
#         super(MyAttacker, self).__init__(model, nnodes, device=device)

#     def attack(self, ori_features, ori_adj, target_node, n_perturbations, **kwargs):
#         pass

class MyAttacker1(BaseAttack):
    def __init__(self, model=None, nnodes=None, device='cpu'):
        super(MyAttacker1, self).__init__(model, nnodes, device=device)

    def attack(self, ori_adj, labels, target_node, n_perturbations):
        # convert the adjacency matrix to a mutable format
        modified_adj = ori_adj.tolil().astype('float')

        # get the degrees of each node
        degrees = ori_adj.sum(axis=1).A1

        # sort nodes by degree, and try to perturb the highest degree nodes first
        nodes_by_degree = np.argsort(degrees)[::-1]

        perturbations = 0

        for node in nodes_by_degree:
            if perturbations == n_perturbations:
                break

            if node == target_node or labels[node] == labels[target_node]:
                continue

            # add an edge from the high-degree node to the target
            if modified_adj[node, target_node] == 0:
                modified_adj[node, target_node] = 1
                modified_adj[target_node, node] = 1
                perturbations += 1
            # if there's already an edge, remove it
            elif modified_adj[node, target_node] == 1:
                modified_adj[node, target_node] = 0
                modified_adj[target_node, node] = 0
                perturbations += 1

        self.modified_adj = modified_adj

        # Return the modified adjacency matrix
        return self.modified_adj.tocsr()