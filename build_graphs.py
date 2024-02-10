import pickle
import torch
from sklearn.model_selection import train_test_split
import dgl


from src.training.masking import add_features
from src.training.utils import weighted_edges, discrete_bin_edges
from src.data.Dataset import FUNSD_loader
from src.data.utils import concat_paragraph2graph_edges
from utils import LoadConfig
from functools import partial

import os

from tqdm import tqdm
import PIL

def vector_func(config, edges):
    node_feat_list = []
    for node_feat in config['features']['node']:
        node_feat = edges.src[node_feat]
        if len(node_feat.size()) == 1:
            node_feat = node_feat.unsqueeze(1)
        node_feat_list.append(node_feat)

    for edge_feat in config['features']['edge']:
        edge_feat = edges.data[edge_feat]
        if len(edge_feat.size()) == 1:
            edge_feat = edge_feat.unsqueeze(1)
        node_feat_list.append(edge_feat)
    
    msg = torch.cat(node_feat_list, dim=1)
    return {'m': msg}

def store_changed_graphs(create_new = False):

    # Storing graphs
    bin_path_train = "/home/nbiescas/Desktop/CVC/CVC_internship/PKL_Graphs/WEIGHTED_CONCAT/train_text_geom_weighted.bin"
    bin_path_val   = "/home/nbiescas/Desktop/CVC/CVC_internship/PKL_Graphs/WEIGHTED_CONCAT/val_text_geom_weighted.bin"
    bin_path_test  = "/home/nbiescas/Desktop/CVC/CVC_internship/PKL_Graphs/WEIGHTED_CONCAT/test_text_geom_weighted.bin"

    # AIXI ES COM HO FEIES ABANS AIXI ES COM HO FEIES ABANS AIXI ES COM HO FEIES ABANS AIXI ES COM HO FEIES ABANS
    #pickle_path_train = '/home/nbiescas/Desktop/CVC/CVC_internship/PKL_Graphs/V2_TEXT_Kmeans_dis_ntresh_discrete/train_TEXT_Kmeans_dis_ntresh_discrete.pkl'
    #pickle_path_validation = '/home/nbiescas/Desktop/CVC/CVC_internship/PKL_Graphs/V2_TEXT_Kmeans_dis_ntresh_discrete/val_TEXT_Kmeans_dis_ntresh_discrete.pkl'
    #pickle_path_test = '/home/nbiescas/Desktop/CVC/CVC_internship/PKL_Graphs/V2_TEXT_Kmeans_dis_ntresh_discrete/test_TEXT_Kmeans_dis_ntresh_discrete.pkl'
    
    if create_new:
        
        data_train = FUNSD_loader(train=True)
        data_test  = FUNSD_loader(train=False)

        def store_features(graphs, desc = 'adding new features train'):
            #graphs = data.graphs
            for id, graph in enumerate(tqdm(graphs, desc=desc)):
                geometric = add_features(graph)
                graph.ndata['area'] = geometric[:, 4]
                graph.ndata['regional_encoding'] = geometric[:, 5:]
                graph.ndata['text_feat'] = graph.ndata['feat'][:, 4:304]
                graph.ndata['Geometric'] = geometric # bounding box, area and region encoding
                # Visual features
                graph.ndata['visual_feat']  = graph.ndata['feat'][:, 304:]
                
                # Distance between nodes
                graph.edata['angle'] = weighted_edges(graph) 
                graph.edata['distance_normalized'] = graph.edata['weights']
                graph.edata['discrete_bin_edges'] = discrete_bin_edges(graph)   
                #graph.apply_edges(concat_paragraph2graph_edges)

                #graph.edata['weights'] = weighted_edges(graph) * graph.edata['weights']
                
            return graphs
        
        data_train, data_val = train_test_split(data_train.graphs, test_size=0.2, random_state=42)


        graphs_train = store_features(data_train, desc='adding new features train')
        graphs_val   = store_features(data_val, desc='adding new features validation')
        graphs_test  = store_features(data_test.graphs, desc='adding new features test')

        # Pretrain model: embedder
        weights_pretrain = '/home/nbiescas/Desktop/CVC/CVC_internship/runs/run109/weights/model_71.pth'
        pretrain_model = torch.load(weights_pretrain).to('cuda:0')
        # Prepare the graphs
        config_pretrain = LoadConfig('run109')
        message_func = partial(vector_func, config_pretrain)

        train_graphs = dgl.batch(graphs_train).to('cuda:0')
        val_graphs = dgl.batch(graphs_val).to('cuda:0')
        test_graphs = dgl.batch(graphs_test).to('cuda:0')

        train_graphs.apply_edges(message_func)
        val_graphs.apply_edges(message_func)
        test_graphs.apply_edges(message_func)

        pretrain_model.eval()
        with torch.no_grad():
            pretrain_embeddings = pretrain_model(train_graphs)
            preval_embeddings = pretrain_model(val_graphs)
            pretest_embeddings = pretrain_model(test_graphs)

            train_graphs.ndata['feat'] = torch.cat((pretrain_embeddings * 0.4,
                                                    train_graphs.ndata['text_feat'] * 0.6),
                                                    #train_graphs.ndata['visual_feat']), 
                                                    dim=1)
            
            val_graphs.ndata['feat'] = torch.cat((preval_embeddings * 0.4,
                                                    val_graphs.ndata['text_feat'] * 0.6),
                                                    #val_graphs.ndata['visual_feat']), 
                                                    dim=1)
            
            test_graphs.ndata['feat'] = torch.cat((pretest_embeddings * 0.4,
                                                    test_graphs.ndata['text_feat'] * 0.6),
                                                    #test_graphs.ndata['visual_feat']), 
                                                    dim=1)

        dgl.save_graphs(bin_path_train, dgl.unbatch(train_graphs))
        dgl.save_graphs(bin_path_val, dgl.unbatch(val_graphs))
        dgl.save_graphs(bin_path_test, dgl.unbatch(test_graphs))

        #graphs_train, graphs_val = train_test_split(graphs_train, test_size=0.2, random_state=42)

        #with open(pickle_path_train, 'wb') as path_train,\
        #    open(pickle_path_validation, 'wb') as path_val,\
        #    open(pickle_path_test, 'wb') as path_test:
#
        #    # Training
        #    pickle.dump(graphs_train, path_train)
        #    # Validation
        #    pickle.dump(graphs_val, path_val)
        #    # Testing
        #    pickle.dump(graphs_test, path_test)

if __name__ == '__main__':
    store_changed_graphs(create_new = True)