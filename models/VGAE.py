import torch
import dgl
from torch import nn
from dgl.nn import GraphConv

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


class VGAE(nn.module):

    def __init__(self, in_feats, out_feats):
        self.GCN        = GraphConv(in_feats,  out_feats, allow_zero_in_degree = True)
        self.mean_GCN   = GraphConv(out_feats, out_feats, allow_zero_in_degree = True)
        self.std_GCN    = GraphConv(out_feats, out_feats, allow_zero_in_degree = True)
        
    def encode(self, graph, feat):
        h1 = self.GCN(graph, feat)
        return self.mean_GCN(graph, h1), self.std_GCN(graph, h1)
    
    def decode(self, z):
        out_matrix = torch.matmul(z, z.T) #The only possitions that matters are either the ones in the upper or lower triangular
        return nn.Sigmoid(out_matrix) #Probabilities
    
    def rpm_trick(self, mean, std):
        """
        Decoder return the a matrix with the scores of the dot product between all the vectors
        
        We need to multiply the latent variables between them and then pass a sigmoid and compare with the original adjacency matrix
        We should sample some edges that contains erase and non erase edges and then observe the score.

        """
        normal = torch.randn(size = mean.shape) #torch.randn generates numbers based on a normal distribution
        z = mean + std * normal
        return z

    def forward(self, graph, feat, edges):
        mean, log_var = self.encode(graph, feat) #Returns two torch.tensors representing the mean and the log variance
        std = log_var.mul(0.5).exp_()
        z = self.rpm_trick(mean, std)  #Z is a matrix (numb_nodes x hidden_dim)
        return self.decode(z)