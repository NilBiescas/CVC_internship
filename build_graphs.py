import pickle
from src.training.masking_funsd import add_features
from src.training.utils import weighted_edges, discrete_bin_edges
from src.data.Data_Loaders import FUNSD_loader
import os
from tqdm import tqdm
import PIL

def store_changed_graphs(create_new = False):

    # Storing graphs
    pickle_path_train = '/home/nbiescas/Desktop/CVC/CVC_internship/PKL_Graphs/train_graph_distance_angle_discrete_bin_edges.pkl'
    pickle_path_test = '/home/nbiescas/Desktop/CVC/CVC_internship/PKL_Graphs/test_graph_distance_angle_discrete_bin_edges.pkl'
    
    if create_new:
        
        data_train = FUNSD_loader(train=True)
        data_test  = FUNSD_loader(train=False)

        def store_features(data, desc = 'adding new features train'):
            graphs = data.graphs
            for id, graph in enumerate(tqdm(graphs, desc=desc)):
                graph.ndata['Geometric'] = add_features(graph)
                # Distance between nodes
                #graph.edata['weights_distance'] = graph.edata['weights']
                # Angles between nodes
                #graph.edata['weights'] = weighted_edges(graph)
                graph.edata['angle'] = weighted_edges(graph)
                graph.edata['distance'] = graph.edata['weights']
                graph.edata['discrete_bin_edges'] = discrete_bin_edges(graph)
                #graph.edata['weights'] = weighted_edges(graph) * graph.edata['weights']
                graph.img_size = PIL.Image.open(data.paths[id]).size
                graph.path = data.paths[id]
            return graphs

        graphs_train = store_features(data_train, desc='adding new features train')
        graphs_test  = store_features(data_test, desc='adding new features test')

        with open(pickle_path_train, 'wb') as path_train, open(pickle_path_test, 'wb') as path_test:
            # Training
            pickle.dump(graphs_train, path_train)
            # Testing
            pickle.dump(graphs_test, path_test)



if __name__ == '__main__':
    #store_changed_graphs(create_new = True)