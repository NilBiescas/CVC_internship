import torch
import dgl
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import torch.nn as nn

class GAT_contrastive(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, num_layers, activation, feat_drop, attn_drop, negative_slope, residual, **kwargs):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        self.num_layers = num_layers
        self.gat_layers.append(GATConv(in_dim, hidden_dim, num_heads, feat_drop, attn_drop, negative_slope, residual, activation=activation))
        for i in range(1, num_layers):
            self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, num_heads, feat_drop, attn_drop, negative_slope, residual, activation=activation))
        self.norm = nn.BatchNorm1d(hidden_dim * num_heads)

        self.fc = nn.Linear(hidden_dim * num_heads, out_features = kwargs['node_classes'])
        

    def forward(self, graph, feat):
        for i in range(self.num_layers):
            feat = self.gat_layers[i](graph, feat).flatten(1)
        feat = self.norm(feat)

        preds = self.fc(feat)
        return preds
    
    def extract_embeddings(self, graphs):
        with torch.no_grad():
            self.eval()
            embeddings = []
            labels = []
            for i in range(self.num_layers):
                feat = self.gat_layers[i](graphs, feat).flatten(1)
            feat = self.norm(feat)
            feat = graphs.ndata['feat'].to('cuda:0')

            raise NotImplementedError
            embeddings.append(self(batch))
            labels.append(label)
            embeddings = torch.cat(embeddings, dim=0)
            labels = torch.cat(labels, dim=0)
            return embeddings.cpu().numpy(), labels.cpu().numpy()







class MaskedGat_contrastive(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, num_layers, activation, feat_drop, attn_drop, negative_slope, residual, **kwargs):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        self.num_layers = num_layers
        self.gat_layers.append(GATConv(in_dim, hidden_dim, num_heads, feat_drop, attn_drop, negative_slope, residual, activation=activation))
        for i in range(1, num_layers):
            self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, num_heads, feat_drop, attn_drop, negative_slope, residual, activation=activation))
        self.norm = nn.BatchNorm1d(hidden_dim * num_heads)

        self.fc = nn.Linear(hidden_dim * num_heads, kwargs['node_classes'])
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))

        kwargs['cfg_edge_predictor']['in_features'] = hidden_dim * num_heads
        self.edge_fc = MLPPredictor_E2E(**kwargs['cfg_edge_predictor'])

    def forward(self, g, x, mask_rate=0.2):
        # ---- attribute reconstruction ----
        preds = self.mask_attr_prediction(g, x, mask_rate=mask_rate)

        return preds
    
    def freeze_network(self, freeze=True):
        # Freeze the network except for the edge_fc
        for name, param in self.named_parameters():
            if 'edge_fc' not in name:
                param.requires_grad = not freeze

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


        for i in range(self.num_layers):
            use_x = self.gat_layers[i](use_g, use_x).flatten(1)

        use_x = self.norm(use_x)

        preds = self.fc(use_x)

        pred_edges = self.edge_fc(use_g, use_x, preds) # 

        return preds, pred_edges
    
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
        # h contains the node representations computed from the GNN z
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.ndata['cls'] = cls
            graph.apply_edges(self.apply_edges)
            return graph.edata['score'] #, graph.edata['box']
        

class MaskedGat_contrastive_linegraphs(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, num_layers, activation, feat_drop, attn_drop, negative_slope, residual, **kwargs):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        self.num_layers = num_layers
        self.gat_layers.append(GATConv(in_dim, hidden_dim, num_heads, feat_drop, attn_drop, negative_slope, residual, activation=activation))
        for i in range(1, num_layers):
            self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, num_heads, feat_drop, attn_drop, negative_slope, residual, activation=activation))
        self.norm = nn.BatchNorm1d(hidden_dim * num_heads)

        self.fc = nn.Linear(hidden_dim * num_heads, kwargs['node_classes'])
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))

    def forward(self, g, x, mask_rate=0.2):
        # ---- attribute reconstruction ----
        preds = self.mask_attr_prediction(g, x, mask_rate=mask_rate)

        return preds
    
    def freeze_network(self, freeze=True):
        # Freeze the network except for the edge_fc
        for name, param in self.named_parameters():
            if 'edge_fc' not in name:
                param.requires_grad = not freeze

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


        for i in range(self.num_layers):
            use_x = self.gat_layers[i](use_g, use_x).flatten(1)

        use_x = self.norm(use_x)

        preds = self.fc(use_x)

        return preds