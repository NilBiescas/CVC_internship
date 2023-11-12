import dgl.function as fn
import torch.nn as nn
import math
import torch
import torch.nn.functional as F

class GcnSAGELayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 Tresh_distance,
                 added_features,
                 bias=True,
                 use_pp=False,
                 use_lynorm=True):
        
        super(GcnSAGELayer, self).__init__()
        #print("in_feats: ->", (2 * in_feats) + added_features)
        #print("out_feats: ->", out_feats)

        self.in_feats = in_feats +  24 # With distance, angle 11
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

class AUTOENCODER_MASK_MODF_SAGE(nn.Module):

    def __init__(self, dimensions_layers, dropout, node_classes, added_features, Tresh_distance, concat_hidden=False, mask_rate=0.2):
        
        super().__init__()
        self._concat_hidden = concat_hidden
        self._mask_rate = mask_rate
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.Tresh_distance = Tresh_distance

        for i in range(len(dimensions_layers) - 1):
            # ENCODER
            self.encoder.append(GcnSAGELayer(   in_feats           = dimensions_layers[i],  
                                                out_feats          = dimensions_layers[i+1],
                                                Tresh_distance     = self.Tresh_distance,
                                                added_features     = added_features,
                                                activation         = F.relu))
            # DECODER
            self.decoder.insert(0, GcnSAGELayer(in_feats           = dimensions_layers[i+1],  
                                                out_feats          = dimensions_layers[i],
                                                added_features     = added_features,
                                                Tresh_distance    = self.Tresh_distance, 
                                                activation=F.relu))
        
        in_dim = dimensions_layers[0]
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))

        m_hidden = dimensions_layers[-1]
        hidden_dim = dimensions_layers[-1]
        if self._concat_hidden:
            m_hidden = sum(dimensions_layers[1:])

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

    def decoder_(self, g, x):
        h = x
        for layer in self.decoder:
            h = layer(g, h)

        return h
    
    def forward(self, g, x, mask_rate=0.2):
        # ---- attribute reconstruction ----
        x_pred, x_true, n_scores = self.mask_attr_prediction(g, x, mask_rate=mask_rate)

        return x_pred, x_true, n_scores
    
    def extract_embeddings(self, graph):
        with torch.no_grad():
            self.eval()
            h = graph.ndata['Geometric'].to('cuda:0')
            h, _ = self.encoder_(graph, h)
            h = h.view(h.shape[0], -1)
            embeddings = h.cpu().detach().numpy()
            labels = graph.ndata['label'].cpu().detach().numpy()
            return embeddings, labels
    
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
        enc_rep, all_hidden = self.encoder_(use_g, use_x)

        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1) # Concat in the column dimensions all the hidden states of the encoder
        
        # Node prediction
        n_scores = self.node_pred(enc_rep)
        #e_scores = self.edge_pred(use_g, enc_rep, n_scores)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        recon = self.decoder_(use_g, rep)

        # Obtain the masked nodes

        x_true = x[mask_nodes]
        x_pred = recon[mask_nodes]

        if mask_rate == 0.0: # If mask rate is 0, we want to reconstruct all the nodes. This is used for validation and test
            x_true = x
            x_pred = recon

        return x_pred, x_true, n_scores