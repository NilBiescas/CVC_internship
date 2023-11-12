import torch
from dgl.nn.pytorch import GraphConv, SAGEConv, GINConv, GATConv
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


# Get the information of the outside edges of each node, sum them, concatenate it with the node bounindg box, and pass it through a MLP # 4 + 9 = 13 -> 10 -> 

class SELF_supervised(nn.Module):

    def __init__(self, dimensions_layers, edge_classes, dropout, edge_pred_features, node_classes, concat_hidden=False, mask_rate=0.2):
        
        super().__init__()
        self._concat_hidden = concat_hidden
        self._mask_rate = mask_rate
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(len(dimensions_layers) - 1): # (11, 8, 6)
            self.encoder.append(SAGEConv(dimensions_layers[i],  dimensions_layers[i+1], aggregator_type = 'pool', activation=F.relu))
            self.decoder.insert(0, SAGEConv(dimensions_layers[i+1],  dimensions_layers[i], aggregator_type = 'pool', activation=F.relu))
        
        #for i in range(len(dimensions_layers) - 1):
        #    self.encoder.append(GATConv(dimensions_layers[i + 1], dimensions_layers[i], num_heads=1, activation=F.relu, allow_zero_in_degree=True))
        #    self.encoder.append(GATConv(dimensions_layers[i + 1], dimensions_layers[i], num_heads=1, activation=F.relu, allow_zero_in_degree=True))
        
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

        # Define edge predictor layer
        #self.edge_pred = MLPPredictor_E2E(m_hidden, hidden_dim, edge_classes, dropout, edge_pred_features)

        # Define node predictor layer
        node_pred = []
        node_pred.append(nn.Linear(hidden_dim, node_classes))
        node_pred.append(nn.LayerNorm(node_classes))
        self.node_pred = nn.Sequential(*node_pred)

    def encoder_(self, g, x):
        h = x
        all_hidden = []
        for conv in self.encoder:
            h = conv(g, h, edge_weight = g.edata['weights'])

            if self._concat_hidden:
                all_hidden.append(h)
        h = self.dropout(h) 
        return h, all_hidden

    def decoder_(self, g, x):
        h = x
        for layer in self.decoder:
            h = layer(g, h, edge_weight = g.edata['weights'])
        
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

        return x_pred, x_true, n_scores


class GAT_masking(nn.Module):

    def __init__(self, dimensions_layers, edge_classes, dropout, edge_pred_features, node_classes, concat_hidden=False, mask_rate=0.2):
        
        super().__init__()
        self._concat_hidden = concat_hidden
        self._mask_rate = mask_rate


        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(len(dimensions_layers) - 1):
            self.encoder.append(GATConv(dimensions_layers[i], dimensions_layers[i + 1], num_heads=1, attn_drop = dropout, activation=F.relu, allow_zero_in_degree=True))
            self.decoder.insert(0, GATConv(dimensions_layers[i + 1], dimensions_layers[i], num_heads=1, attn_drop = dropout, activation=F.relu, allow_zero_in_degree=True))
        
        in_dim = dimensions_layers[0]
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))

        print("\nEncoder: {} \tNumb Layers: {}".format(self.encoder.__repr__(), len(dimensions_layers)))
        print("\nDecoder: {} \tNumb Layers: {}".format(self.decoder.__repr__(), len(dimensions_layers)))

        m_hidden = dimensions_layers[-1]
        hidden_dim = dimensions_layers[-1]
        if self._concat_hidden:
            m_hidden = sum(dimensions_layers[1:])

        if concat_hidden:
            print("Concatenating hidden states")
            self.encoder_to_decoder = nn.Linear(m_hidden, hidden_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Define edge predictor layer
        #self.edge_pred = MLPPredictor_E2E(m_hidden, hidden_dim, edge_classes, dropout, edge_pred_features)

        # Define node predictor layer
        node_pred = []
        node_pred.append(nn.Linear(hidden_dim, node_classes))
        node_pred.append(nn.LayerNorm(node_classes))
        self.node_pred = nn.Sequential(*node_pred)

    def encoder_(self, g, h):
        all_hidden = []
        for conv in self.encoder:
            h = conv(g, h)

            if self._concat_hidden:
                all_hidden.append(h)

        return h, all_hidden

    def decoder_(self, g, x):
        h = x
        for layer in self.decoder:
            h = layer(g, h)
        return h
    
    def extract_embeddings(self, graph):
        with torch.no_grad():
            self.eval()
            h = graph.ndata['Geometric'].to('cuda:0')
            h, _ = self.encoder_(graph, h)
            h = h.view(h.shape[0], -1)
            embeddings = h.cpu().detach().numpy()
            labels = graph.ndata['label'].cpu().detach().numpy()
            return embeddings, labels
        
    def forward(self, g, x, mask_rate = 0.2):
        # ---- attribute reconstruction ----
        x_pred, x_true, n_scores, mask_nodes = self.mask_attr_prediction(g, x, mask_rate=mask_rate)

        return x_pred, x_true, n_scores, mask_nodes
    
    def encoding_mask_noise(self, g, x, mask_rate=0.2):
        if mask_rate == 0.0:
            return g, x, (0, None)
        
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
    
    def mask_attr_prediction(self, g, x, mask_rate=0.2):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, mask_rate) # Mask the features
 
        use_g = pre_use_g
        enc_rep, all_hidden = self.encoder_(use_g, use_x)

        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1) # Concat in the column dimensions all the hidden states of the encoder
        
        # Node prediction in the masked nodes
        #nodes_to_predict = enc_rep[mask_nodes]
        n_scores = self.node_pred(enc_rep)

        #e_scores = self.edge_pred(use_g, enc_rep, n_scores)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        recon = self.decoder_(use_g, rep)

        # Obtain the masked nodes
        x_pred = recon[mask_nodes]
        x_true = x[mask_nodes]

        return x_pred, x_true, n_scores, mask_nodes
 

class MLPPredictor_E2E(nn.Module):
    def __init__(self, in_features, hidden_dim, out_classes, dropout,  edge_pred_features):
        super().__init__()
        self.out = out_classes
        self.W1 = nn.Linear(in_features * 2 +  edge_pred_features, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.W2 = nn.Linear(hidden_dim, out_classes)
        self.drop = nn.Dropout(dropout)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        cls_u = F.softmax(edges.src['cls'], dim=1)
        cls_v = F.softmax(edges.dst['cls'], dim=1)
        polar = edges.data['feat']

        x = F.relu(self.norm(self.W1(torch.cat((h_u, cls_u, polar, h_v, cls_v), dim=1))))
        score = self.drop(self.W2(x))

        return {'score': score}

    def forward(self, graph, h, cls):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.ndata['cls'] = cls
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']