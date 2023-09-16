from d_preprocess import load_npz_file, read_letter, random_remove_edges
from torch.utils.data import DataLoader
import dgl
from dgl import DropEdge
from torch.nn.utils.rnn import pad_sequence
import functools

from training import device

def collate(graphs):
    """
    Function that returns a graph with random deletion edges and the original adjacency matrix
    """
    transform = DropEdge(p=0.4)
    #partial_removed_edges = functools.partial(dgl.transforms.DropEdge(), p=0.4)
    removed_edges = list(map(transform, graphs))
    batched_graph = dgl.batch(removed_edges)
    adjacency_matrices_target = [graph.adj().to_dense().to(device) for graph in graphs]
    #adjacency_matrix = batched_graph.adj().to_dense() #Obtaining the adjacency matrix for the big batch graph. The model needs to not connect nodes of different graphs
    return batched_graph, adjacency_matrices_target

    #sparse_adj = list(map(lambda x: x.adj().to_dense().view(-1), graphs))
    #partial_removed_edges = functools.partial(random_remove_edges, prob=0.2)
    #removed_edges = list(map(partial_removed_edges, graphs))
    #batched_graph = dgl.batch(removed_edges)
    #return batched_graph, pad_sequence(sparse_adj, batch_first=True, padding_value=-1)

def z_normalization(graph):
    features = graph.ndata.pop('feat')
    mean = features.mean(dim=0)
    std = features.std(dim=0)
    graph.ndata['feat'] = (features - mean) / std
    return graph

def loaders(DATA_PATH):
    transform = dgl.transforms.RowFeatNormalizer(subtract_min=True, node_feat_names=['feat'])
    trainset, validset, testset = load_npz_file(DATA_PATH)

    Train_Graphs = [transform(dgl.to_float(read_letter(graph))) for graph in trainset]
    Valid_Graphs = [transform(dgl.to_float(read_letter(graph))) for graph in validset]
    Test_Graphs  = [transform(dgl.to_float(read_letter(graph))) for graph in testset]

    train_loader    = DataLoader(Train_Graphs, batch_size=64, shuffle = True, collate_fn=collate)
    val_loader      = DataLoader(Valid_Graphs, batch_size=64, collate_fn=collate)
    test_loader     = DataLoader(Test_Graphs,  batch_size=32, collate_fn=collate)
    return train_loader, val_loader, test_loader