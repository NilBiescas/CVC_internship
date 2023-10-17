from ..models.VGAE import GAE, GSage_AE, GIN_AE, GAT_AE
from ..models.Drop_edge import E2E

from ..models.VGAE import device
from ..models.SELF_AEC import SELF_supervised
import math
from ..data.doc2_graph.utils import get_config
from sklearn.utils import class_weight
import numpy as np
import torch
import wandb

from typing import Tuple
import pickle
import dgl

from sklearn.model_selection import train_test_split
from typing import Tuple

def discrete_positions(rect_src : list, rect_dst : list) -> Tuple[int, int]:
    """Compute distance and angle from src to dst bounding boxes (poolar coordinates considering the src as the center)
    Args:
        rect_src (list) : source rectangle coordinates
        rect_dst (list) : destination rectangle coordinates
    
    Returns:
        tuple (ints): distance and angle
    """
    
    # check relative position
    left = (rect_dst[2] - rect_src[0]) <= 0
    bottom = (rect_src[3] - rect_dst[1]) <= 0
    right = (rect_src[2] - rect_dst[0]) <= 0
    top = (rect_dst[3] - rect_src[1]) <= 0
    
    vp_intersect = (rect_src[0] <= rect_dst[2] and rect_dst[0] <= rect_src[2]) # True if two rects "see" each other vertically, above or under
    hp_intersect = (rect_src[1] <= rect_dst[3] and rect_dst[1] <= rect_src[3]) # True if two rects "see" each other horizontally, right or left
    rect_intersect = vp_intersect and hp_intersect
    
    if rect_intersect:
        return 0 #'intersect'
    elif top and left:
        return 1 #'top_left'
    elif left and bottom:
        return 2 #'bottom_left'
    elif bottom and right:
        return 3 #'bottom_right'
    elif right and top:
        return 4 #'top_right'
    elif left:
        return 5 #'left'
    elif right:
        return 6 #'right'
    elif bottom:
        return 7 #'bottom'
    elif top:
        return 8 #'top'  
    
    #number2_position = {0:'intersect', 1:'top_left', 2:'bottom_left', 3:'bottom_right', 4:'top_right', 5:'left', 6:'right', 7:'bottom', 8:'top'}

def load_graphs(data, path, load=False):
    if load:
        print("SAVING ")
        train_graphs, val_graphs, _, _ = train_test_split(data.graphs, torch.ones(len(data.graphs), 1), test_size=0.2, random_state=42)
        print("-> Number of training graphs: ", len(train_graphs))
        print("-> Number of validation graphs: ", len(val_graphs))

        #Graph for training
        train_graphs = get_relative_positons(train_graphs)
        train_graph = dgl.batch(train_graphs)
        train_graph = train_graph.int()

        #Graph for validating
        val_graphs = get_relative_positons(val_graphs)
        val_graph = dgl.batch(val_graphs)
        val_graph = val_graph.int().to(device)

        with open ('/home/nbiescas/Desktop/CVC/CVC_internship/src/train_graph.pkl', 'wb') as train:
            pickle.dump(train_graph, train)

        with open('/home/nbiescas/Desktop/CVC/CVC_internship/src/val_graph.pkl', 'wb') as val:
            pickle.dump(val_graph, val)
        


    print("LOADING GRAPHS FROM MEMORY")
    with open("/home/nbiescas/Desktop/CVC/CVC_internship/src/train_graph.pkl", 'rb') as train:
        train_graph = pickle.load(train)
    with open("/home/nbiescas/Desktop/CVC/CVC_internship/src/val_graph.pkl", 'rb') as val:
        val_graph   = pickle.load(val)

    return train_graph, val_graph


def get_relative_positons(data):

    for graph in data.graphs:
        src, dst = graph.edges()
        discret_info = []
        for src_idx, dst_idx in zip(src, dst):
            src_idx = src_idx.item()
            dst_idx = dst_idx.item()
            relative_position = discrete_positions(graph.nodes[src_idx][0]['geom'][0], graph.nodes[dst_idx][0]['geom'][0])
            discret_info.append(relative_position)
        graph.edata['discrete_info'] = torch.tensor(discret_info)
    return data

def log_wandb(mode, recons_loss, node_reconstruction_loss, graph_reconstructin_loss):

    print(f"\n{'- '*10}{mode}{' -'*10}\n")
    wandb.log({"test loss": recons_loss, "Test node reconstruction loss": node_reconstruction_loss, "Test graph reconstructin loss": graph_reconstructin_loss})
    print('\nFeature reconstruction loss test: {:.6f}'.format(recons_loss))
    print('Node reconstruction loss test:  {}\tGraph reconstruction loss test: {}'.format(node_reconstruction_loss, graph_reconstructin_loss))

def compute_crossentropy_loss(scores : torch.Tensor, labels : torch.Tensor):
    w = class_weight.compute_class_weight(class_weight='balanced', classes= np.unique(labels.cpu().numpy()), y=labels.cpu().numpy())
    return torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to('cuda:0'))(scores, labels)


def get_optimizer(model, config):
    if config.optim == 'SGD': 
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9)
    elif config.optim == 'ADAMW':
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

def get_model(config, data):
    #Dimensions of the autencoder
    layers_dimensions = config.layers_dimensions # e.g (100, 25, 10)

    if config.model == 'SAGE':
        model = GSage_AE(layers_dimensions).to(device)
    elif config.model == 'GAE':
        model = GAE(layers_dimensions).to(device)
    elif config.model == 'GIN':
        model = GIN_AE(layers_dimensions).to(device)
    elif config.model == 'GAT':
        model = GAT_AE(layers_dimensions).to(device)
    elif config.model == 'SELF':
        model = SELF_supervised(layers_dimensions).to(device)
    elif config.model == 'E2E':
        edge_pred_features = int((math.log2(get_config('preprocessing').FEATURES.num_polar_bins) + data.node_num_classes)*2)
        model = E2E(node_classes = data.node_num_classes, 
                    edge_classes = data.edge_num_classes, 
                    dimensions_layers = layers_dimensions, 
                    dropout=0.2, 
                    edge_pred_features=edge_pred_features,
                    drop_rate = config.drop_rate,
                    discrete_pos = config.discrete,
                    bounding_box = config.bbox
                    ).to(device)
    
    return model