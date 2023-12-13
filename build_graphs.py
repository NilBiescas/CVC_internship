import pickle
import torch

from src.training.masking import add_features
from src.training.utils import weighted_edges, discrete_bin_edges
from src.data.Dataset import FUNSD_loader
from src.data.utils import concat_paragraph2graph_edges
import os

from tqdm import tqdm
import PIL

def message_func(config, edges):

    node_feat_list = []
    for node_feat in config['node_feat']:
        node_feat = edges.src[node_feat]
        if len(node_feat.size()) == 1:
            node_feat = node_feat.unsqueeze(1)
        node_feat_list.append(node_feat)

    for edge_feat in config['edge_feat']:
        edge_feat = edges.data[edge_feat]
        if len(edge_feat.size()) == 1:
            edge_feat = edge_feat.unsqueeze(1)
        node_feat_list.append(edge_feat)
    
    msg = torch.cat(node_feat_list, dim=1)
    return {'m': msg}

def store_changed_graphs(create_new = False):

    # Storing graphs
    pickle_path_train = '/home/nbiescas/Desktop/CVC/CVC_internship/PKL_Graphs/kmeans_contrastive/train_kmeans_contrastive.pkl'
    pickle_path_test = '/home/nbiescas/Desktop/CVC/CVC_internship/PKL_Graphs/kmeans_contrastive/test_kmeans_contrastive.pkl'
    
    
    if create_new:
        
        data_train = FUNSD_loader(train=True)
        data_test  = FUNSD_loader(train=False)

        def store_features(data, desc = 'adding new features train'):
            graphs = data.graphs
            for id, graph in enumerate(tqdm(graphs, desc=desc)):
                graph.ndata['Geometric'] = add_features(graph) # bounding box, are region encoding
                # Distance between nodes
                graph.edata['angle'] = weighted_edges(graph) 
                graph.edata['distance_normalized'] = graph.edata['weights']
                graph.edata['edges_discret'] = discrete_bin_edges(graph)
                graph.apply_edges(concat_paragraph2graph_edges)

                #graph.edata['weights'] = weighted_edges(graph) * graph.edata['weights']
                
            return graphs

        graphs_train = store_features(data_train, desc='adding new features train')
        graphs_test  = store_features(data_test, desc='adding new features test')

        with open(pickle_path_train, 'wb') as path_train, open(pickle_path_test, 'wb') as path_test:
            # Training
            pickle.dump(graphs_train, path_train)
            # Testing
            pickle.dump(graphs_test, path_test)


if __name__ == '__main__':
    store_changed_graphs(create_new = True)