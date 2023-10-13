import torch
import dgl

from dgl.nn.pytorch import GraphConv, SAGEConv, GINConv, GATConv
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def drop_edge(graph, drop_rate):
    """
    Returns the graph with the edges dropped and the positions of the edges dropped. 
    The edges dropped do not belong to the set of added edges (edges that do not exist in the original graph)
    """

    real_edges = graph.edata['label'].nonzero().squeeze(1)
    E = len(real_edges)

    mask_rates = torch.FloatTensor(np.ones(E) * drop_rate)
    masks = torch.bernoulli(mask_rates)
    mask_idx = masks.nonzero().squeeze(1)

    pos = real_edges[mask_idx]

    return dgl.remove_edges(graph, pos.type(torch.int32)), pos

class E2E(nn.Module):
    def __init__(self, node_classes, 
                       edge_classes,
                       dimensions_layers,
                       dropout,
                       edge_pred_features,
                       drop_rate,
                       doProject=True):

        super().__init__()
        
        self.drop_rate = drop_rate
        
        # Perform message passing
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(len(dimensions_layers) - 1):
            self.encoder.append(SAGEConv(dimensions_layers[i],  dimensions_layers[i+1], aggregator_type = 'pool', activation=F.relu))
            self.decoder.insert(0, SAGEConv(dimensions_layers[i+1],  dimensions_layers[i], aggregator_type = 'pool', activation=F.relu))

        # Define edge predictor layer
        m_hidden = dimensions_layers[-1]
        hidden_dim = dimensions_layers[-1]

        self.edge_pred = MLPPredictor_E2E(m_hidden, hidden_dim, edge_classes, dropout, edge_pred_features)

        # Define node predictor layer
        node_pred = []
        node_pred.append(nn.Linear(m_hidden, node_classes))
        node_pred.append(nn.LayerNorm(node_classes))
        self.node_pred = nn.Sequential(*node_pred)

        # Define bounding box predictor
        bbox_coordinates = 4
        bounding_box_pred = []
        bounding_box_pred.append(nn.Linear(m_hidden, bbox_coordinates))
        bounding_box_pred.append(nn.LayerNorm(bbox_coordinates))
        self.bbox = nn.Sequential(*bounding_box_pred)

    def forward(self, g, h):
        if self.drop_rate > 0:
            new_g, _ = drop_edge(g, self.drop_rate)
            for layer in self.encoder:
                h = layer(new_g, h)
        else:
            for layer in self.encoder:
                h = layer(g, h)

        node_pred = self.node_pred(h) # Node prediction, given the features of the latent space, returns a vector with the unbound logits for each class
        edges_pred = self.edge_pred(g, h, node_pred) # Edge prediction

        for layer in self.decoder:
            h = layer(g, h)

        return node_pred, edges_pred, h

class MLPPredictor_E2E(nn.Module):
    def __init__(self, in_features, hidden_dim, out_classes, dropout,  edge_pred_features):
        super().__init__()
        self.out = out_classes
        self.W1 = nn.Linear(in_features * 2 +  edge_pred_features, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.W2 = nn.Linear(hidden_dim, out_classes)
        self.drop = nn.Dropout(dropout)
        #self.W3 = nn.Linear(hidden_dim, 4)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        cls_u = F.softmax(edges.src['cls'], dim=1)
        cls_v = F.softmax(edges.dst['cls'], dim=1)
        polar = edges.data['feat']

        x = F.relu(self.norm(self.W1(torch.cat((h_u, cls_u, polar, h_v, cls_v), dim=1))))
        score = self.drop(self.W2(x))

        #bounding_box = self.W3(x)
        return {'score': score} #, 'box': bounding_box}

    def forward(self, graph, h, cls):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.ndata['cls'] = cls
            graph.apply_edges(self.apply_edges)
            return graph.edata['score'] #, graph.edata['box']