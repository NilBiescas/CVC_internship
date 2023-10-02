from ..data.d_preprocess import load_npz_file, read_letter, random_remove_edges
from torch.utils.data import DataLoader
import dgl
from dgl import DropEdge
from torch.nn.utils.rnn import pad_sequence
import pprint

from .doc2_graph.utils import get_config
from ..models.VGAE import device

def collate(graphs):
    """
    Function that returns a graph with random deletion edges and the original adjacency matrix
    """
    transform = DropEdge(p=0.1)
    removed_edges = list(map(transform, graphs))
    batched_graph = dgl.batch(removed_edges)
    adjacency_matrices_target = [graph.adj().to_dense().to(device) for graph in graphs]
    return batched_graph, adjacency_matrices_target

def z_normalization(graph):
    features = graph.ndata.pop('feat')
    mean = features.mean(dim=0)
    std = features.std(dim=0)
    graph.ndata['feat'] = (features - mean) / std
    return graph

def OMNIGLOT_loader(DATA_PATH):
    transform = dgl.transforms.RowFeatNormalizer(subtract_min=True, node_feat_names=['feat'])
    trainset, validset, testset = load_npz_file(DATA_PATH)

    Train_Graphs = [transform(dgl.to_float(read_letter(graph))) for graph in trainset]
    Valid_Graphs = [transform(dgl.to_float(read_letter(graph))) for graph in validset]
    Test_Graphs  = [transform(dgl.to_float(read_letter(graph))) for graph in testset]

    train_loader    = DataLoader(Train_Graphs, batch_size=64, shuffle = True, collate_fn=collate)
    val_loader      = DataLoader(Valid_Graphs, batch_size=64, collate_fn=collate)
    test_loader     = DataLoader(Test_Graphs,  batch_size=32, collate_fn=collate)
    return train_loader, val_loader, test_loader


from .doc2_graph.data.dataloader import Document2Graph
from .doc2_graph.paths import FUNSD_TRAIN, TRAIN_SAMPLES


def FUNSD_loader(train = True):
    config = get_config('preprocessing')

    pprint.pprint(config, indent=4, width=1)
    print("\n")
    if train:
        return Document2Graph(name='FUNSD TRAIN', src_path=FUNSD_TRAIN, device = "cuda:0", output_dir=TRAIN_SAMPLES)
    else:
        return Document2Graph(name='FUNSD TEST', src_path=FUNSD_TRAIN, device = "cuda:0", output_dir=TRAIN_SAMPLES)
