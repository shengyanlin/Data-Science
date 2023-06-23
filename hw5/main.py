from argparse import ArgumentParser
from core import GCN
import torch
from numpy import ndarray
from scipy import sparse as sp
from attacker import MyAttacker1
from core import Judge


# DO NOT modify
def get_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, default='target_nodes_list.txt', help='Target node file.')
    parser.add_argument('--data_path', type=str, default='./data/data.pkl', help='Input graph.')
    parser.add_argument('--model_path', type=str, default='saved-models/gcn.pt', help='GNN model to attack.')
    parser.add_argument('--use_gpu', action='store_true')
    return parser.parse_args()


# # DO NOT modify this function signature
def attack(adj: sp.csr_matrix, features: sp.csr_matrix, labels: ndarray,
           idx_train: ndarray, idx_val: ndarray, idx_test: ndarray,
           target_node: int, n_perturbations: int, **kwargs) -> sp.spmatrix:
    """
    Args:
        adj (sp.csr_matrix): Original (unperturbed) adjacency matrix
        features (sp.csr_matrix): Original (unperturbed) node feature matrix
        labels (ndarray): node labels
        idx_train (ndarray):node training indices
        idx_val (ndarray): node validation indices
        idx_test (ndarray): node test indices
        target_node (int): target node index to be attacked
        n_perturbations (int): Number of perturbations on the input graph.

    Returns:
        sp.spmatrix: attacked (perturbed) adjacency matrix
    """
    # TODO: Setup your attack model
    # print(f'other args: {kwargs}')
    
    # # TODO: Setup your attack model
    # model = MyAttacker2(surrogate, nnodes=adj.shape[0], device=device)
    # model = model.to(device)
    # modified_adj = model.attack(features, adj, labels, idx_train, target_node, n_perturbations)
    
    # ##=========Attacker1=========##
    model = MyAttacker1()
    model = model.to(device)
    modified_adj = model.attack(adj, labels, target_node, n_perturbations)
    
    return modified_adj


if __name__ == '__main__':
    args = get_args()

    cuda = torch.cuda.is_available()
    print(f'cuda: {cuda}')
    device = torch.device("cuda" if cuda else "cpu")
    
    judge = Judge(args.data_path, args.model_path, args.input_file, device=device)
    judge.multi_test(attack, device=device)

    # You can pass other arguments like this
    # judge.multi_test(attack, arg1='foo', args2='bar')

    # FYI: You can load the data like this
    # data = pickle.load(Path(args.data_path).open('rb'))
    # adj, features, labels, idx_train, idx_val, idx_test = data

    # FYI: setup the surrogate model
    # from core import GCN
    # surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16, dropout=0, with_relu=False, with_bias=False, device=device)
    # surrogate = surrogate.to(device)
    # surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)



