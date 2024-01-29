import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GraphConv
import math
import dgl.function as fn
import torch.nn.functional as F

class EdgeApplyModule(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.batch_norm = nn.BatchNorm1d(out_dim)
        self.activation = activation
    
    def forward(self, edges):
        m = torch.cat((edges.src['h'], edges.data['m']), dim=1)
        m = self.linear(m)
        m = self.batch_norm(m)
        m = self.activation(m)
        return {'m': m}


class GNN_edges(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,):
        
        super(GNN_edges, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.activation = activation
        self.apply_edges = EdgeApplyModule(2 * in_feats, out_feats, activation).to('cuda:0')

    def forward(self, g, m): 
        g.edata['m'] = m
        g.send_and_recv(g.edges(), fn.copy_e('m', 's'), fn.sum('s', 'h'))
        g.apply_edges(self.apply_edges)
        return g.edata.pop('m'), g.ndata.pop('h')


class Simple_edge_encoder(nn.Module):
    def __init__(self, layers_dimensions, activation, **kwargs):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.layers_dimensions = layers_dimensions

        for dim in range(len(self.layers_dimensions) - 1):
            self.encoder.append(GNN_edges(self.layers_dimensions[dim], self.layers_dimensions[dim + 1], activation=activation))

    def forward(self, graph):
        feat_edges = graph.edata['m']
        for layer in self.encoder:
            feat_edges, feat_nodes = layer(graph, feat_edges)
        return feat_nodes
    
    def get_embeddigns(self, loader):
        import numpy as np
        with torch.no_grad():
            embeddings = []
            labels = []
            for graph, label in loader:
                graph = graph.to('cuda:0')
                out = self.forward(graph)
                embeddings.append(out.cpu().numpy())
                labels.append(label.cpu().numpy())

            embeddings = np.concatenate(embeddings, axis=0)
            labels = np.concatenate(labels, axis=0)
            return embeddings, labels


def lynorm(x):
    return x

class Contrastive_model_edges_features(nn.Module):
    def __init__(self, layers_dimensions, activation, in_dim, out_dim, use_lynorm, norm, dropout, **kwargs):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.layers_dimensions = layers_dimensions

        self.l1 = nn.Linear(in_dim, out_dim, bias = True)
        self.dropout = nn.Dropout(dropout)

        self.user_norm = kwargs.get('user_norm', False)
        
        self.sequential = nn.Sequential(nn.Linear(in_dim, out_dim, bias = True),
                                        nn.ReLU(),
                                        nn.Linear(out_dim, out_dim, bias = True),
                                        nn.ReLU(),
                                        nn.Linear(out_dim, out_dim, bias = True))
        if use_lynorm:
            self.lynorm = nn.LayerNorm(out_dim, elementwise_affine=True)
        else:
            self.lynorm = lynorm

        for dim in range(len(self.layers_dimensions) - 1):
            self.encoder.append(
                GraphConv(layers_dimensions[dim], layers_dimensions[dim + 1], 
                          norm=norm, weight=True, bias=True, 
                          activation=activation, allow_zero_in_degree=True))
    
    def message_func(self, edges):
        out_feat = self.sequential(edges.data['m'])
        out_feat = self.lynorm(out_feat)
        #out_feat = self.lynorm(self.l1(edges.data['m']))
        return {'new_m': out_feat}
    
    def forward(self, graph):
        graph.send_and_recv(graph.edges(), self.message_func, fn.sum('new_m', 'h'))
        feat_nodes = graph.ndata.pop('h')
        if self.user_norm:
            feat_nodes = feat_nodes * graph.ndata['norm']
        for layer in self.encoder:
            feat_nodes = layer(graph, feat_nodes)
        feat_nodes = self.dropout(feat_nodes)
        return feat_nodes
    
    def get_embeddigns(self, loader):
        import numpy as np
        with torch.no_grad():
            embeddings = []
            labels = []
            for graph, label in loader:
                graph = graph.to('cuda:0')
                out = self.forward(graph)
                embeddings.append(out.cpu().numpy())
                labels.append(label.cpu().numpy())

            embeddings = np.concatenate(embeddings, axis=0)
            labels = np.concatenate(labels, axis=0)
            return embeddings, labels
        
class GcnSAGELayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 Tresh_distance,
                 bias=True,
                 use_pp=False,
                 use_lynorm=True):
        
        super(GcnSAGELayer, self).__init__()
        #print("in_feats: ->", (2 * in_feats) + added_features)
        #print("out_feats: ->", out_feats)

        self.in_feats = in_feats + 24 # With distance, angle 11
        self.linear = nn.Linear(self.in_feats, out_feats, bias=bias)
        self.activation = activation
        self.use_pp = use_pp
        self.Tresh_distance = Tresh_distance

        if use_lynorm:
            self.lynorm = nn.LayerNorm(out_feats, elementwise_affine=True)
        else:
            self.lynorm = lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)

    # Define the message function
    def message_func(self, edges):
        # Get the features of node1
        h_src = edges.src['Geometric']

        # Get the features of the edge
        distance = edges.data['distance']
        angle    = edges.data['angle']
        discrete = edges.data['discrete_bin_edges']
        bin_angles = edges.data['feat']

        distance = distance.unsqueeze(1)
        angle    = angle.unsqueeze(1)

        msg = torch.cat((h_src, distance, angle, bin_angles, discrete), dim=1)
        return {'m': msg}

    # Define the reduce function
    def reduce_func(self, nodes):
        # Sum the messages received by each node
        return {'h': torch.sum(nodes.mailbox['m'], dim=1)}

    def forward(self, g, h): # h comes from g.ndata['Geometric']
        g = g.local_var()
        
        #g.ndata['Geometric'] = h
        norm = g.ndata['norm']

        if not self.Tresh_distance:
            g.edata['distance'] = g.edata['distance_not_tresh']
        
        g.send_and_recv(g.edges(), self.message_func, self.reduce_func)
        ah = g.ndata.pop('h')
        h = self.concat(h, ah, norm)

        h = self.linear(h)
        h = self.lynorm(h)
        if self.activation:
            h = self.activation(h)
        return h

    def concat(self, h, ah, norm):
        ah = ah * norm
        h = torch.cat((h, ah), dim=1)
        return h

class GCN_LAYER(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 Tresh_distance,
                 added_features = 24,
                 bias=True,
                 use_pp=False,
                 use_lynorm=True):
        
        super(GCN_LAYER, self).__init__()

        self.added_features = added_features
        self.in_feats = in_feats + self.added_features #+ 24 # With distance, angle 11
        self.linear = nn.Linear(self.in_feats, out_feats, bias=bias)
        self.activation = activation
        self.use_pp = use_pp
        self.Tresh_distance = Tresh_distance

        if use_lynorm:
            self.lynorm = nn.LayerNorm(out_feats, elementwise_affine=True)
        else:
            self.lynorm = lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h):
        g = g.local_var()
        
        norm = g.ndata['norm']

        g.send_and_recv(g.edges(), fn.copy_e('m', 'h'), fn.sum('h', 'sum_h'))
        ah = g.ndata.pop('sum_h')
        h = self.concat(h, ah, norm)

        h = self.linear(h)
        h = self.lynorm(h)
        if self.activation:
            h = self.activation(h)
        return h

    def concat(self, h, ah, norm):
        ah = ah * norm
        h = torch.cat((h, ah), dim=1)
        return h

class simplified_gcn_contrastive_model(nn.Module):
    def __init__(self, layers_dimensions, dropout, Tresh_distance, added_features = 24, concat_hidden=False, **kwargs):
        super(simplified_gcn_contrastive_model, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers_dimensions = layers_dimensions
        self.added_features = added_features
        self.norm = kwargs.get('norm', True)
        layers = []
        for i in range(len(self.layers_dimensions) - 1):
            layers.append(nn.Linear(self.layers_dimensions[i] + self.added_features, self.layers_dimensions[i+1], bias=True))
            layers.append(nn.ReLU())

        self.linear_projections = nn.Sequential(*layers)
    
    def project_message(self, edges):
        m = edges.data['m']
        m = self.message_passing_projection(m)
        return {'m': m}
    
    def forward(self, g, h):
        g = g.local_var()
        norm = g.ndata['norm']
        g.apply_edges(self.project_message)
        g.update_all(fn.copy_e('m', 'h'), fn.sum('h', 'sum_h'))
        ah = g.ndata.pop('sum_h')
        h = self.concat(h, ah, norm)
        h = self.linear_projections(h)
        h = self.dropout(h)
        return h
    
    def concat(self, h, ah, norm):
        if self.norm:
            ah = ah * norm
        h = torch.cat((h, ah), dim=1)
        return h

class simplified_contrastive_model(nn.Module):
    def __init__(self, layers_dimensions, dropout, Tresh_distance, added_features = 24, concat_hidden=False, **kwargs):
        super(simplified_contrastive_model, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers_dimensions = layers_dimensions
        self.added_features = added_features
        self.norm = kwargs.get('norm', True)
        layers = []
        for i in range(len(self.layers_dimensions) - 1):
            layers.append(nn.Linear(self.layers_dimensions[i] + self.added_features, self.layers_dimensions[i+1], bias=True))
            layers.append(nn.ReLU())

        self.linear_projections = nn.Sequential(*layers)
    
    def forward(self, g, h):
        g = g.local_var()
        norm = g.ndata['norm']
        g.send_and_recv(g.edges(), fn.copy_e('m', 'h'), fn.sum('h', 'sum_h'))
        ah = g.ndata.pop('sum_h')
        h = self.concat(h, ah, norm)
        h = self.linear_projections(h)
        h = self.dropout(h)
        return h

    def concat(self, h, ah, norm):
        if self.norm:
            ah = ah * norm
        h = torch.cat((h, ah), dim=1)
        return h

class class_contrastive_model(nn.Module):
    def __init__(self, layers_dimensions, dropout, Tresh_distance, added_features = 24, concat_hidden=False, **kwargs):
        
        super(class_contrastive_model, self).__init__()
        self._concat_hidden = concat_hidden
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList()
        self.Tresh_distance = Tresh_distance
        self.layers_dimensions = layers_dimensions
        
        for i in range(len(layers_dimensions) - 1):
            # ENCODER
            self.encoder.append(GCN_LAYER(   in_feats           = layers_dimensions[i],  
                                                out_feats          = layers_dimensions[i+1],
                                                Tresh_distance     = self.Tresh_distance,
                                                activation         = F.relu,
                                                added_features     = added_features))
    def forward(self, g):
        h = g.ndata['Geometric']
        all_hidden = []
        for conv in self.encoder:
            h = conv(g, h)

            if self._concat_hidden:
                all_hidden.append(h)
                
        h = self.dropout(h) 
        return h

    def get_embeddigns(self, loader):
        import numpy as np
        with torch.no_grad():
            embeddings = []
            labels = []
            for graph, label in loader:
                graph = graph.to('cuda:0')
                x = graph.ndata['Geometric'].to('cuda:0')
                out = self.forward(graph)
                embeddings.append(out.cpu().numpy())
                labels.append(label.cpu().numpy())

            embeddings = np.concatenate(embeddings, axis=0)
            labels = np.concatenate(labels, axis=0)
            return embeddings, labels
        

class AUTOENCODER_MASK_MODF_SAGE_CONTRASTIVE(nn.Module):
    def __init__(self, layers_dimensions, dropout, node_classes, Tresh_distance, concat_hidden=False, mask_rate=0.2, **kwargs):
        
        super().__init__()
        self._concat_hidden = concat_hidden
        self._mask_rate = mask_rate
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList()
        self.Tresh_distance = Tresh_distance
        self.layers_dimensions = layers_dimensions
        
        for i in range(len(layers_dimensions) - 1):
            # ENCODER
            self.encoder.append(GcnSAGELayer(   in_feats           = layers_dimensions[i],  
                                                out_feats          = layers_dimensions[i+1],
                                                Tresh_distance     = self.Tresh_distance,
                                                activation         = F.relu))
        
        in_dim = layers_dimensions[0]
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))

        m_hidden = layers_dimensions[-1]
        hidden_dim = layers_dimensions[-1]
        if self._concat_hidden:
            m_hidden = sum(layers_dimensions[1:])

        if concat_hidden:
            print("Concatenating hidden states")
            self.encoder_to_decoder = nn.Linear(m_hidden, hidden_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Define node predictor layer
        node_pred = []
        node_pred.append(nn.Linear(hidden_dim, node_classes))
        node_pred.append(nn.LayerNorm(node_classes))
        self.node_pred = nn.Sequential(*node_pred)

    def encoder_(self, g, x):
        h = x
        all_hidden = []
        for conv in self.encoder:
            h = conv(g, h)

            if self._concat_hidden:
                all_hidden.append(h)
                
        h = self.dropout(h) 
        return h, all_hidden

    
    def forward(self, g):
        # ---- attribute reconstruction ----
        x = g.ndata['Geometric']
        embeddings = self.mask_attr_prediction(g, x, mask_rate=self._mask_rate)

        return embeddings
    
    def get_embeddigns(self, loader):
        import numpy as np
        with torch.no_grad():
            embeddings = []
            labels = []
            for graph, label in loader:
                graph = graph.to('cuda:0')
                x = graph.ndata['Geometric'].to('cuda:0')
                out, _ = self.encoder_(graph, x)
                embeddings.append(out.cpu().numpy())
                labels.append(label.cpu().numpy())

            embeddings = np.concatenate(embeddings, axis=0)
            labels = np.concatenate(labels, axis=0)
            return embeddings, labels
        
    #def extract_embeddings(self, graph, features = None):
    #    with torch.no_grad():
    #        self.eval()
    #        # Get the features of the nodes
    #        h = graph.ndata['Geometric'].to('cuda:0') if features is None else features.to('cuda:0')
    #        
    #        h, _ = self.encoder_(graph, h)
    #        h = h.view(h.shape[0], -1)
    #        embeddings = h.cpu().detach().numpy()
    #        labels = graph.ndata['label'].cpu().detach().numpy()
    #        return embeddings, labels
    
    def encoding_mask_noise(self, g, x, mask_rate=0.2):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device) #shuffle the nodes of the whole graph

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0 # The mask is 0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)
    
    def mask_attr_prediction(self, g, x, mask_rate):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, mask_rate) # Mask the features
 
        use_g = pre_use_g
        enc_rep, _ = self.encoder_(use_g, use_x)

        return enc_rep