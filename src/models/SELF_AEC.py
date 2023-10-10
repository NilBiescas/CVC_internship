import torch
from dgl.nn.pytorch import GraphConv, SAGEConv, GINConv, GATConv
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class SELF_supervised(nn.Module):

    def __init__(self, dimensions_layers):

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(len(dimensions_layers) - 1):
            self.encoder.append(SAGEConv(dimensions_layers[i],  dimensions_layers[i+1], aggregator_type = 'pool', activation=F.relu))
            self.decoder.insert(0, SAGEConv(dimensions_layers[i+1],  dimensions_layers[i], aggregator_type = 'pool', activation=F.relu))

        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)
        
        in_dim = dimensions_layers[0]
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))

    def forward(self, g, x):
        # ---- attribute reconstruction ----
        x_pred, x_true = self.mask_attr_prediction(g, x)

        return x_pred, x_true
    
    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device) #shuffle the nodes of the whole graph
        num_mask_nodes = int(mask_rate * num_nodes) # How many nodes to mask

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
    
    def mask_attr_prediction(self, g, x):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate) # Mask the features
 
        use_g = pre_use_g
        enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)

        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1) # Concat in the column dimensions all the hidden states of the encoder

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        recon = self.decoder(pre_use_g, rep)

        # Obtain the masked nodes
        x_true = x[mask_nodes]
        x_pred = recon[mask_nodes]

        return x_pred, x_true