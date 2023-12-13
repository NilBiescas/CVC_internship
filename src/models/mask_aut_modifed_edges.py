import dgl.function as fn
import torch.nn as nn
import math
import torch
import torch.nn.functional as F

class V2_GcnSAGELayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,):
        
        super(V2_GcnSAGELayer, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.activation = activation
        self.linear_node = nn.Linear(in_feats, out_feats, bias=True)
        self.linear_edge = nn.Linear(2 * in_feats, out_feats, bias=True)

        self.reset_parameters(self.linear_node)
        self.reset_parameters(self.linear_edge)

    def reset_parameters(self, linear):
        stdv = 1. / math.sqrt(linear.weight.size(1))
        linear.weight.data.uniform_(-stdv, stdv)
        if linear.bias is not None:
            linear.bias.data.uniform_(-stdv, stdv)

    def f_edge(self, edges):
        return {'m2': torch.cat((edges.src['h'], edges.data['m']), dim=1)}
        #return {'m2': edges.src['h'] + edges.data['m']}

    def forward(self, g, m): 
        with g.local_scope():
            g.edata['m'] = m
            g.send_and_recv(g.edges(), fn.copy_e('m', 's'), fn.sum('s', 'h'))
            g.apply_edges(self.f_edge)

            m = g.edata.pop('m2')
            h = g.ndata.pop('h')

            #m = torch.cat((m, m2), dim = 1)
            m = self.linear_edge(m)
            h = self.linear_node(h)

            return self.activation(m), self.activation(h)

    def concat(self, h, ah, norm):
        ah = ah * norm
        h = torch.cat((h, ah), dim=1)
        return h

class V2_AUTOENCODER_MASK_MODF_SAGE(nn.Module):

    def __init__(self, dimensions_layers, dropout, node_classes, concat_hidden=False, mask_rate=0.2, activation=F.relu, **kwargs):
        super().__init__()
        self._concat_hidden = concat_hidden
        self._mask_rate = mask_rate
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.activation = activation
        for i in range(len(dimensions_layers) - 1):
            # ENCODER
            self.encoder.append(V2_GcnSAGELayer(in_feats           = dimensions_layers[i],  
                                                out_feats          = dimensions_layers[i+1],
                                                activation         = self.activation))
            # DECODER
            self.decoder.insert(0, V2_GcnSAGELayer(in_feats        = dimensions_layers[i+1],  
                                                   out_feats       = dimensions_layers[i],
                                                   activation      = self.activation))
            
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

    def encoder_(self, g, m): # m comes from edata
        all_hidden = []
        for conv in self.encoder:
            m, h = conv(g, m) # Update the edge features
            m = self.dropout(m)
        
        return m, h, all_hidden

    def decoder_(self, g, m):
        for conv in self.decoder:
            m, _ = conv(g, m)
        return m
    
    def forward(self, g, m, mask_rate=0.2):
        # ---- attribute reconstruction ----
        x_pred, x_true, n_scores = self.mask_attr_prediction(g, m, mask_rate=mask_rate)

        return x_pred, x_true, n_scores
    
    def extract_embeddings(self, graph):
        with torch.no_grad():
            self.eval()
            m = graph.edata['m'].to('cuda:0')
            _, h, _ = self.encoder_(graph, m)
            h = h.view(h.shape[0], -1)
            embeddings = h.cpu().detach().numpy()
            labels = graph.ndata['label'].cpu().detach().numpy()
            return embeddings, labels
    
    def encoding_mask_noise(self, g, e, mask_rate=0.2):
        num_edges = g.num_edges()
        perm = torch.randperm(num_edges, device=e.device) #shuffle the nodes of the whole graph

        # random masking
        num_mask_edges = int(mask_rate * num_edges)
        mask_edges = perm[: num_mask_edges]
        keep_edges = perm[num_mask_edges: ]

        out_x = e.clone()
        token_nodes = mask_edges
        out_x[mask_edges] = 0.0 # The mask is 0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_edges, keep_edges)
    
    def mask_attr_prediction(self, g, m, mask_rate):
        pre_use_g, use_x, (mask_edges, keep_edges) = self.encoding_mask_noise(g, m, mask_rate) # Mask the features
 
        use_g = pre_use_g
        m1, h, all_hidden = self.encoder_(use_g, use_x)

        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1) # Concat in the column dimensions all the hidden states of the encoder
        
        # Node prediction
        n_scores = self.node_pred(h)
        #e_scores = self.edge_pred(use_g, enc_rep, n_scores)

        # ---- attribute reconstruction ----
        m2 = self.encoder_to_decoder(m1)

        recon = self.decoder_(use_g, m2) # Edge features

        # Obtain the masked nodes

        x_pred = recon[mask_edges]
        x_true = m[mask_edges]

        if mask_rate == 0.0: # If mask rate is 0, we want to reconstruct all the nodes. This is used for validation and test
            x_true = m
            x_pred = recon
        
        return x_pred, x_true, n_scores