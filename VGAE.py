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
    def __init__(self, inpu_size, hidden_dim, output_size):
        super(VGAEModel, self).__init__()
        self.inpu_size = inpu_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        layers = [GraphConv(self.inpu_size, self.hidden_dim, activation=F.relu, norm='right',allow_zero_in_degree=True),
                  GraphConv(self.hidden_dim, self.hidden_dim, activation=F.relu, allow_zero_in_degree=True),
                  GraphConv(self.hidden_dim, int(self.hidden_dim + self.hidden_dim / 2), activation=F.relu, allow_zero_in_degree=True),
                  GraphConv(int(self.hidden_dim + self.hidden_dim / 2), int(self.hidden_dim * 2), activation=F.relu, allow_zero_in_degree=True),
                  GraphConv(int(self.hidden_dim * 2), int(self.hidden_dim * 2), activation=F.relu, allow_zero_in_degree=True),
                  GraphConv(int(self.hidden_dim * 2), self.output_size, activation=lambda x: x, allow_zero_in_degree=True),
                  GraphConv(int(self.hidden_dim * 2), self.output_size, activation=lambda x: x, allow_zero_in_degree=True)]
        self.layers = nn.ModuleList(layers)

    def encoder(self, g, h):
        for layer in self.layers[:-2]:
            h = layer(g, h)
        self.mean = self.layers[-2](g, h)
        self.log_std = self.layers[-1](g, h) #
        gaussian_noise = torch.randn(h.size(0), self.output_size).to(device)
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std.mul(0.5)).to(device) #The think that changes is that he is using the log_std insted of the log_var
        return sampled_z

    def decoder(self, g, z): # The positions of the features are ordered in the way they where put it in the list
        start_idx = 0
        output_matrix = []
        for nodes in g.batch_num_nodes():
            end_index = start_idx + nodes
            features_graph = z[start_idx:end_index]
            output_matrix.append(torch.sigmoid(torch.matmul(features_graph, features_graph.t())))
            start_idx = end_index
        return output_matrix
        # bg.batch_num_nodes returns the number of nodes for each graph [matrix list]


    def forward(self, g, features):
        z = self.encoder(g, features) # z = number_nodes x dimensions
        adj_rec = self.decoder(g, z)

        start_idx = 0
        means_list, log_std_list = [], []
        for nodes in g.batch_num_nodes():
            end_index = start_idx + nodes
            means_list.append(self.mean[start_idx:end_index])
            log_std_list.append(self.log_std[start_idx:end_index])
            start_idx = end_index

        return adj_rec, means_list, log_std_list


class VGAE(nn.Module):

    def __init__(self, in_feats, out_feats):
        super(VGAE, self).__init__()

        self.GCN        = GraphConv(in_feats,  out_feats, allow_zero_in_degree = True)
        self.mean_GCN   = GraphConv(out_feats, out_feats, allow_zero_in_degree = True)
        self.std_GCN    = GraphConv(out_feats, out_feats, allow_zero_in_degree = True)
        
    def encode(self, graph, feat):
        h1 = self.GCN(graph, feat)
        return self.drself.mean_GCN(graph, h1), self.std_GCN(graph, h1)
    
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