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


def log_wandb(mode, recons_loss, node_reconstruction_loss, graph_reconstructin_loss):

    print(f"\n{'- '*10}{mode}{' -'*10}\n")
    wandb.log({"test loss": recons_loss, "Test node reconstruction loss": node_reconstruction_loss, "Test graph reconstructin loss": graph_reconstructin_loss})
    print('\nFeature reconstruction loss test: {:.6f}'.format(recons_loss))
    print('Node reconstruction loss test:  {}\tGraph reconstruction loss test: {}'.format(node_reconstruction_loss, graph_reconstructin_loss))

def compute_crossentropy_loss(scores : torch.Tensor, labels : torch.Tensor):
    w = class_weight.compute_class_weight(class_weight='balanced', classes= np.unique(labels.cpu().numpy()), y=labels.cpu().numpy())
    return torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to('cuda:0'))(scores, labels)

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
                    drop_rate = config.drop_rate).to(device)
    
    return model