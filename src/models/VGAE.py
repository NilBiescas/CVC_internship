import torch
from dgl.nn.pytorch import GraphConv, SAGEConv, GINConv, GATConv
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VGAEModel(nn.Module):
    def __init__(self, inpu_size, hidden_dim, output_size):
        super(VGAEModel, self).__init__()
        self.inpu_size = inpu_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        layers = [GraphConv(self.inpu_size, self.hidden_dim, activation=F.relu, allow_zero_in_degree=True),
                    GraphConv(self.hidden_dim, self.output_size, activation=lambda x: x, allow_zero_in_degree=True),
                    GraphConv(self.hidden_dim, self.output_size, activation=lambda x: x, allow_zero_in_degree=True)]
        self.layers = nn.ModuleList(layers)

    def encoder(self, g, h):
        h = self.layers[0](g, h)
        self.mean = self.layers[1](g, h)
        self.log_std = self.layers[2](g, h)
        gaussian_noise = torch.randn(h.size(0), self.output_size).to(device)
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std).to(device) #The think that changes is that he is using the log_std insted of the log_var
        return sampled_z

    def decoder(self, g, z): # The positions of the features are ordered in the way they where put it in the list
        start_idx = 0
        output_matrix = []
        for nodes in g.batch_num_nodes():
            end_index = start_idx + nodes
            features_graph = z[start_idx:end_index]
            output_matrix.append(torch.sigmoid(torch.matmul(features_graph, features_graph.t())))
            start_idx = end_index
        return output_matrix
        # bg.batch_num_nodes returns the number of nodes for each graph [matrix list]


    def forward(self, g, features):
        z = self.encoder(g, features) # z = number_nodes x dimensions
        adj_rec = self.decoder(g, z)

        start_idx = 0
        means_list, log_std_list = [], []
        for nodes in g.batch_num_nodes():
            end_index = start_idx + nodes
            means_list.append(self.mean[start_idx:end_index])
            log_std_list.append(self.log_std[start_idx:end_index])
            start_idx = end_index

        return adj_rec, means_list, log_std_list


class GSage_AE(nn.Module):

    def __init__(self, dimensions_layers):
        super(GSage_AE, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(len(dimensions_layers) - 1):
            self.encoder.append(SAGEConv(dimensions_layers[i],  dimensions_layers[i+1], aggregator_type = 'pool', activation=F.relu))
            self.decoder.insert(0, SAGEConv(dimensions_layers[i+1],  dimensions_layers[i], aggregator_type = 'pool', activation=F.relu))

        print("\nEncoder: {} \tNumb Layers: {}".format(self.encoder.__repr__(), len(dimensions_layers)))
        print("\nDecoder: {} \tNumb Layers: {}".format(self.decoder.__repr__(), len(dimensions_layers)))


    def forward(self, graph, feat):
        for layer in self.encoder:
            feat = layer(graph, feat, edge_weight = graph.edata['weights'])

        for layer in self.decoder:
            feat = layer(graph, feat, edge_weight = graph.edata['weights'])

        return feat

class GAT_AE(nn.Module):

    def __init__(self, dimensions_layers):
        super(GAT_AE, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(len(dimensions_layers) - 1):
            self.encoder.append(GINConv(torch.nn.Linear(dimensions_layers[i], dimensions_layers[i + 1]),    aggregator_type = 'max', activation=F.relu))
            if i == 0:
                self.decoder.insert(0, GATConv(dimensions_layers[i + 1], dimensions_layers[i], num_heads=1, activation=F.relu, allow_zero_in_degree=True))
            else:
                self.decoder.insert(0, GINConv(torch.nn.Linear(dimensions_layers[i + 1], dimensions_layers[i]), aggregator_type = 'max', activation=F.relu)) # GAT(in_dim, out_dim) GAT out_dim * 3

        
        print("\nEncoder: {} \tNumb Layers: {}".format(self.encoder.__repr__(), len(dimensions_layers)))
        print("\nDecoder: {} \tNumb Layers: {}".format(self.decoder.__repr__(), len(dimensions_layers)))


    def forward(self, graph, feat):
        for layer in self.encoder:
            feat = layer(graph, feat)

        for layer in self.decoder:
            feat = layer(graph, feat)
        return feat


class GIN_AE(nn.Module):

    def __init__(self, dimensions_layers):
        super(GIN_AE, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(len(dimensions_layers) - 1):
            self.encoder.append(GINConv(torch.nn.Linear(dimensions_layers[i], dimensions_layers[i + 1]),    aggregator_type = 'max', activation=F.relu))
            self.decoder.insert(0, GINConv(torch.nn.Linear(dimensions_layers[i + 1], dimensions_layers[i]), aggregator_type = 'max', activation=F.relu))

        print("\nEncoder: {} \tNumb Layers: {}".format(self.encoder.__repr__(), len(dimensions_layers)))
        print("\nDecoder: {} \tNumb Layers: {}".format(self.decoder.__repr__(), len(dimensions_layers)))

    def forward(self, graph, feat):
        for layer in self.encoder:
            feat = layer(graph, feat, edge_weight = graph.edata['weights'])

        for layer in self.decoder:
            feat = layer(graph, feat, edge_weight = graph.edata['weights'])

        return feat


class GAE(nn.Module):

    def __init__(self, dimensions_layers):
        super(GAE, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(len(dimensions_layers) - 1):
            self.encoder.append(GraphConv(dimensions_layers[i],  dimensions_layers[i+1], allow_zero_in_degree = True))
            self.decoder.insert(0, GraphConv(dimensions_layers[i+1],  dimensions_layers[i], allow_zero_in_degree = True))

        print("\nEncoder: {} \tNumb Layers: {}".format(self.encoder.__repr__(), len(dimensions_layers)))
        print("\nDecoder: {} \tNumb Layers: {}".format(self.decoder.__repr__(), len(dimensions_layers)))


    def forward(self, graph, feat):
        for layer in self.encoder:
            feat = layer(graph, feat, edge_weight = graph.edata['weights'])

        for layer in self.decoder:
            feat = layer(graph, feat, edge_weight = graph.edata['weights'])

        return feat

class GSage_AE(nn.Module):

    def __init__(self, dimensions_layers):
        super(GSage_AE, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(len(dimensions_layers) - 1):
            self.encoder.append(SAGEConv(dimensions_layers[i],  dimensions_layers[i+1], aggregator_type = 'pool', activation=F.relu))
            self.decoder.insert(0, SAGEConv(dimensions_layers[i+1],  dimensions_layers[i], aggregator_type = 'pool', activation=F.relu))

        print("\nEncoder: {} \tNumb Layers: {}".format(self.encoder.__repr__(), len(dimensions_layers)))
        print("\nDecoder: {} \tNumb Layers: {}".format(self.decoder.__repr__(), len(dimensions_layers)))

class E2E(nn.Module):
    def __init__(self, node_classes, 
                       edge_classes,
                       dimensions_layers,
                       dropout,
                       edge_pred_features,
                       doProject=True):

        super().__init__()

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

    def forward(self, g, h):
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