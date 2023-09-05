import numpy as np
import torch
import dgl

def load_npz_file(filename):
    if (filename.split('.')[-1] != 'npz'):
        raise ValueError("Invalid file format")
    load_data = np.load(filename, allow_pickle=True, encoding='latin1')
    return load_data['train'], load_data['valid'], load_data['test']

def read_letter(data, self_loops = True):
    """
    Takes a numpy matrix encoding the nodes of the graph.
    """
    if self_loops:
        adj_matrix = np.identity(len(data)) 
    else:
        adj_matrix = np.zeros((len(data), len(data))) 
    node_feat  = np.zeros((len(data), 2)) # features (num_nodes x 2) 2 because the features are the cordinates of each node    
    
    x = 0
    y = 0
    for node_id, row in enumerate(data):
        x += row[0]
        y -= row[1]
        node_feat[node_id] = [x, y]
        if (node_id != 0):
            _, _, previous_lift = data[node_id - 1]
            if (previous_lift == 1):
                continue
            adj_matrix[node_id][node_id - 1] = 1
            adj_matrix[node_id - 1][node_id] = 1

    src, dst = np.nonzero(adj_matrix)
    g = dgl.graph((src, dst))
    g.ndata['feat'] = torch.from_numpy(node_feat)

    #G = nx.from_numpy_array(am) # Createa a graph from an adjacency matrix
    #nx.set_node_attributes(G, node_label, 'position')

    return g