import dgl
import torch
from typing import List

TRAIN_CONTRASTIVE = '/home/nbiescas/Desktop/CVC/CVC_internship/runs/run42/graphs_contrastive/train_contrastive.bin'
TEST_CONTRASTIVE = '/home/nbiescas/Desktop/CVC/CVC_internship/runs/run42/graphs_contrastive/test_contrastive.bin'

def new_labels(graphs : List[dgl.DGLGraph], line_graphs : List[dgl.DGLGraph]):
    for graph, line_graph in zip(graphs, line_graphs):
        node_labels = graph.ndata['label']
        line_graph.ndata['label'] = torch.tensor([1 if node_labels[v].item() == 3 and node_labels[u].item() == 0 else 0 for v, u in zip(*graph.edges())])
        line_graph.ndata['feat'] = line_graph.ndata['m']
    return line_graphs

if __name__ == '__main__':
    # Load the train and test graphs with the contrastive features obtained using the contrastive learning model
    train, _ = dgl.load_graphs(TRAIN_CONTRASTIVE)
    test, _ = dgl.load_graphs(TEST_CONTRASTIVE)

    # Obtain the line graphs of the train and test graphs
    line_graphs_train   = [dgl.line_graph(g, shared = True) for g in train]
    line_graphs_test    = [dgl.line_graph(g, shared = True) for g in test]

    # Add the new labels to the line graphs
    line_graphs_train   = new_labels(train, line_graphs_train)
    line_graphs_test    = new_labels(test, line_graphs_test)

    path_train = '/home/nbiescas/Desktop/CVC/CVC_internship/PKL_Graphs/line_graphs_kmeans_contrastive/train_line_graphs.bin'
    path_test = '/home/nbiescas/Desktop/CVC/CVC_internship/PKL_Graphs/line_graphs_kmeans_contrastive/test_line_graphs.bin'

    dgl.save_graphs(path_train, line_graphs_train)
    dgl.save_graphs(path_test, line_graphs_test)