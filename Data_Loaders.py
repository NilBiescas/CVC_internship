from d_preprocess import load_npz_file, read_letter, random_remove_edges
from torch.utils.data import DataLoader
import dgl


def collate(graphs):
    """
    Function that returns a graph with random deletion edges and the original adjacency matrix
    """
    sparse_adj = list(map(lambda x: x.adj(), graphs))
    removed_edges = list(map(random_remove_edges, graphs))
    batched_graph = dgl.batch(removed_edges)
    return batched_graph, sparse_adj



def loaders(DATA_PATH):
    trainset, validset, testset = load_npz_file(DATA_PATH)

    Train_Graphs = [read_letter(graph) for graph in trainset[:10]]
    Valid_Graphs = [read_letter(graph) for graph in validset[:10]]
    Test_Graphs  = [read_letter(graph) for graph in testset[:10]]

    train_loader    = DataLoader(Train_Graphs, batch_size=1, shuffle = True, collate_fn=collate)
    val_loader      = DataLoader(Valid_Graphs, batch_size=1, collate_fn=collate)
    test_loader     = DataLoader(Test_Graphs, batch_size=1, collate_fn=collate)
    return train_loader, val_loader, test_loader