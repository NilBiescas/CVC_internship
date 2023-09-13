import torch
from dgl.nn.pytorch import GraphConv
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

TODO = """
Features of the nodes: Relative Position [x, y]
Input = Perturbed dgl.graph with missing edges.

Implementation of VGAE https://arxiv.org/abs/1611.07308

The idea of the Encoder in the VGAE is:

-   Do one step of message passing using a Graph convolutions netwrok (GCN)
-   Project using two different weight matrices the features to represents the means and the std.


-   matrix of the mean vectors
-   matrix of log variances.

"""

from training import device

class VGAEModel(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim):
        super(VGAEModel, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim

        layers = [GraphConv(self.in_dim, self.hidden1_dim, activation=F.relu, allow_zero_in_degree=True),
                  GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True),
                  GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True)]
        self.layers = nn.ModuleList(layers)

    def encoder(self, g, features):
        h = self.layers[0](g, features)
        self.mean = self.layers[1](g, h)
        self.log_std = self.layers[2](g, h)
        gaussian_noise = torch.randn(features.size(0), self.hidden2_dim).to(device)
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std).to(device) #The think that changes is that he is using the log_std insted of the log_var
        return sampled_z

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, g, features):
        z = self.encoder(g, features)
        adj_rec = self.decoder(z)
        return adj_rec


class VGAE(nn.Module):

    def __init__(self, in_feats, out_feats):
        super(VGAE, self).__init__()

        self.GCN        = GraphConv(in_feats,  out_feats, allow_zero_in_degree = True)
        self.mean_GCN   = GraphConv(out_feats, out_feats, allow_zero_in_degree = True)
        self.std_GCN    = GraphConv(out_feats, out_feats, allow_zero_in_degree = True)
        
    def encode(self, graph, feat):
        h1 = self.GCN(graph, feat)
        return self.mean_GCN(graph, h1), self.std_GCN(graph, h1)
    
    def reparametrize(self, mu, logvar):
        """
        Decoder return the a matrix with the scores of the dot product between all the vectors
        
        We need to multiply the latent variables between them and then pass a sigmoid and compare with the original adjacency matrix
        We should sample some edges that contains erase and non erase edges and then observe the score.

        """
        std = logvar.exp_()
        #std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_().cpu()
        eps = Parameter(eps)
        return eps.mul(std).add_(mu) # Scale-location transformation of N(0, 1)
    
    def decode(self, z):
        out_matrix = torch.sigmoid(torch.matmul(z, z.t())) #The only possitions that matters are either the ones in the upper or lower triangular
        return out_matrix #Probabilities
    
    def forward(self, graph, feat):
        mu, logvar = self.encode(graph, feat) #Returns two torch.tensors representing the mean and the log variance
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar