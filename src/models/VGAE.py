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