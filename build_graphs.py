import pickle
import torch
from sklearn.model_selection import train_test_split

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
    pickle_path_train = '/home/nbiescas/Desktop/CVC/CVC_internship/PKL_Graphs/V2_TEXT_Kmeans_dis_ntresh_discrete/train_TEXT_Kmeans_dis_ntresh_discrete.pkl'
    pickle_path_validation = '/home/nbiescas/Desktop/CVC/CVC_internship/PKL_Graphs/V2_TEXT_Kmeans_dis_ntresh_discrete/val_TEXT_Kmeans_dis_ntresh_discrete.pkl'
    pickle_path_test = '/home/nbiescas/Desktop/CVC/CVC_internship/PKL_Graphs/V2_TEXT_Kmeans_dis_ntresh_discrete/test_TEXT_Kmeans_dis_ntresh_discrete.pkl'

    if create_new:
        
        data_train = FUNSD_loader(train=True)
        data_test  = FUNSD_loader(train=False)

        def store_features(data, desc = 'adding new features train'):
            graphs = data.graphs
            for id, graph in enumerate(tqdm(graphs, desc=desc)):
                geometric = add_features(graph)
                graph.ndata['area'] = geometric[:, 4]
                graph.ndata['regional_encoding'] = geometric[:, 5:]
                graph.ndata['text_feat'] = graph.ndata['feat'][:, 4:]
                graph.ndata['Geometric'] = geometric # bounding box, area and region encoding

                # Distance between nodes
                graph.edata['angle'] = weighted_edges(graph) 
                graph.edata['distance_normalized'] = graph.edata['weights']
                graph.edata['discrete_bin_edges'] = discrete_bin_edges(graph)   
                graph.apply_edges(concat_paragraph2graph_edges)

                #graph.edata['weights'] = weighted_edges(graph) * graph.edata['weights']
                
            return graphs

        graphs_train = store_features(data_train, desc='adding new features train')
        graphs_test  = store_features(data_test, desc='adding new features test')


        graphs_train, graphs_val = train_test_split(graphs_train, test_size=0.2, random_state=42)

        with open(pickle_path_train, 'wb') as path_train,\
            open(pickle_path_validation, 'wb') as path_val,\
            open(pickle_path_test, 'wb') as path_test:

            # Training
            pickle.dump(graphs_train, path_train)
            # Validation
            pickle.dump(graphs_val, path_val)
            # Testing
            pickle.dump(graphs_test, path_test)

if __name__ == '__main__':
    store_changed_graphs(create_new = True)