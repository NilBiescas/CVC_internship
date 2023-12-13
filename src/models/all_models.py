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

    def forward(self, g, x, mask_rate=0.2):
        # ---- attribute reconstruction ----
        preds = self.mask_attr_prediction(g, x, mask_rate=mask_rate)

        return preds
    
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