import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import torch.nn as nn
from src.data.doc2_graph.models.unet.model import Unet
from PIL import Image
import torchvision
import torchvision.transforms.functional as tvF
from pathlib import Path

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
        
class UNET_MaskedGat_contrastive_UNET(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, num_layers, activation, feat_drop, attn_drop, negative_slope, residual, **kwargs):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        self.num_layers = num_layers
        self.gat_layers.append(GATConv(in_dim, hidden_dim, num_heads, feat_drop, attn_drop, negative_slope, residual, activation=activation, allow_zero_in_degree=True))
        for i in range(1, num_layers):
            self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, num_heads, feat_drop, attn_drop, negative_slope, residual, activation=activation, allow_zero_in_degree=True))
        self.norm = nn.BatchNorm1d(hidden_dim * num_heads)
        
        if kwargs.get('layers_dimensions', None) is None:
            self.fc = nn.Linear(hidden_dim * num_heads, kwargs['node_classes'])
        else:
            dimensions_layers = kwargs['layers_dimensions']
            layers = []
            for i in range(len(dimensions_layers) - 1):
                layers.append(nn.Linear(dimensions_layers[i], dimensions_layers[i + 1]))
            layers.append(nn.Linear(dimensions_layers[-1], kwargs['node_classes']))
            self.fc = nn.Sequential(*layers)
        
        # He cambait the in_dim a
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))

        kwargs['cfg_edge_predictor']['in_features'] = hidden_dim * num_heads
        self.edge_fc = MLPPredictor_E2E(**kwargs['cfg_edge_predictor'])

        self.visual_embedder = Unet(encoder_name="mobilenet_v2", encoder_weights=None, in_channels=1, classes=4)
         
        CHECKPOINTS = Path('/home/nbiescas/Desktop/CVC/CVC_internship/src/data/doc2_graph/models/checkpoints')
        self.visual_embedder.load_state_dict(torch.load(CHECKPOINTS / 'backbone_unet.pth')['weights'])
        self.visual_embedder = self.visual_embedder.encoder
        self.visual_embedder.to('cuda:0')
        self.device = 'cuda:0'
        
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
    
    def UNET_forward(self, g):
        inter_vectors = []
        batch_num_nodes = g.batch_num_nodes()
        start = 0
        for id, path in enumerate(g.imgs):
            bounding_boxes = g.ndata['geom'][start: start + batch_num_nodes[id],:]
            size = Image.open(path).size
            img = Image.open(path)
            visual_emb = self.visual_embedder(tvF.to_tensor(img).unsqueeze_(0).to(self.device)) # output [batch, channels, dim1, dim2]
            bboxs = [torch.Tensor(b) for b in  bounding_boxes]
            bboxs = [torch.stack(bboxs, dim=0).to(self.device)]
            h = [torchvision.ops.roi_align(input=ve, boxes=bboxs, spatial_scale=1/ min(size[1] / ve.shape[2] , size[0] / ve.shape[3]), output_size=1) for ve in visual_emb[1:]]
            h = torch.cat(h, dim=1)
            inter_vectors.append(h.view(batch_num_nodes[id], -1))
            start += batch_num_nodes[id]
            
        out = torch.cat(inter_vectors, dim=0)
        
        out = torch.cat((g.ndata['feat'], out), dim=1)
        return out
        
    def mask_attr_prediction(self, g, x, mask_rate):
        x = self.UNET_forward(g)
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, mask_rate) # Mask the features
        
        use_g = pre_use_g
        for i in range(self.num_layers):
            use_x = self.gat_layers[i](use_g, use_x).flatten(1)

        use_x = self.norm(use_x)
        preds = self.fc(use_x)
        pred_edges = self.edge_fc(use_g, use_x, preds)
        return preds, pred_edges

class MaskedGat_contrastive(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, num_layers, activation, feat_drop, attn_drop, negative_slope, residual, **kwargs):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        self.num_layers = num_layers
        self.gat_layers.append(GATConv(in_dim, hidden_dim, num_heads, feat_drop, attn_drop, negative_slope, residual, activation=activation))
        for i in range(1, num_layers):
            self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, num_heads, feat_drop, attn_drop, negative_slope, residual, activation=activation))
        self.norm = nn.BatchNorm1d(hidden_dim * num_heads)
        
        if kwargs.get('layers_dimensions', None) is None:
            self.fc = nn.Linear(hidden_dim * num_heads, kwargs['node_classes'])
        else:
            dimensions_layers = kwargs['layers_dimensions']
            layers = []
            for i in range(len(dimensions_layers) - 1):
                layers.append(nn.Linear(dimensions_layers[i], dimensions_layers[i + 1]))
            layers.append(nn.Linear(dimensions_layers[-1], kwargs['node_classes']))
            self.fc = nn.Sequential(*layers)
        
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
        pred_edges = self.edge_fc(use_g, use_x, preds)
        return preds, pred_edges
    
class MLPPredictor_E2E(nn.Module):
    def __init__(self, in_features, hidden_dim, out_classes, dropout, edge_pred_features, layers_dimensions = None):
        super().__init__()
        self.out = out_classes
        self.in_feat = in_features * 2 + edge_pred_features

        if layers_dimensions is not None:
            layers = []
            for i in range(len(layers_dimensions) - 1):
                layers.append(nn.Linear(layers_dimensions[i], layers_dimensions[i + 1]))
            layers.append(nn.Linear(layers_dimensions[-1], hidden_dim))
            self.W1 = nn.Sequential(*layers)
        else:
            self.W1 = nn.Linear(in_features * 2 + edge_pred_features, hidden_dim)
  
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
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.ndata['cls'] = cls
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']