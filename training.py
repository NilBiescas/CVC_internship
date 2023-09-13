import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F 
from IPython import display
import matplotlib.pyplot as plt

#Implementation of the loss: https://github.com/JuliaSun623/VGAE_dgl/blob/main/train.py. Later on implement the norm parameter


device = torch.device("cuda:0")

def loss_VGAE(logits, adj, mean, log_std):

    loss =  F.binary_cross_entropy(logits.view(-1), adj.view(-1))
    kl_divergence = 0.5 / logits.size(0) * (
                1 + 2 * log_std - mean ** 2 - torch.exp(log_std) ** 2).sum(
            1).mean()
    loss -= kl_divergence
    return loss

@torch.no_grad()  # prevent this function from computing gradients see https://pytorch.org/docs/stable/generated/torch.no_grad.html
def validate(criterion, model, loader):

    val_loss = 0
    correct = 0

    model.eval()

    for graph, sparse_adj in loader:
        sparse_adj = sparse_adj[0]
        if (torch.cuda.is_available()):
            graph, sparse_adj = graph.to(device), sparse_adj.to(device)
            features = graph.ndata.pop('feat').to(device)
        else:
            features = graph.ndata.pop('feat')

        adjacency_matrix = sparse_adj.to_dense()
        output = model(graph, features)
        loss = criterion(output, adjacency_matrix, output.mean, output.log_std)
        val_loss += loss.item()

        correct += torch.sum(output == adjacency_matrix)                                                           

    total_edges = 0
    for graph, _ in loader:
        total_edges += graph.number_of_edges()

    val_loss /= len(loader.dataset)
    accuracy = 100. * correct / total_edges
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, total_edges,
        accuracy))
    
    return val_loss

def train(epoch, criterion, model, optimizer, loader):
    
    total_loss = 0.0

    model.train()

    for batch_idx, (graph, sparse_adj) in enumerate(loader):
        sparse_adj = sparse_adj[0]
        optimizer.zero_grad()
        if (torch.cuda.is_available()):
            graph, sparse_adj = graph.to(device), sparse_adj.to(device)
            features = graph.ndata.pop('feat').to(device)
        else:
            features = graph.ndata.pop('feat')

        adjacency_matrix = sparse_adj.to_dense()
        output = model(graph, features)
        loss = criterion(output, adjacency_matrix, output.mean, output.log_std)
        loss.backward()
        optimizer.step()
        
        # print loss every N iterations
        if batch_idx % 100 == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(
                epoch, loss.item()))


        total_loss += loss.item()  #.item() is very important here? Why? To sum up the indivuals losses of each of the samples seen in an epoch. Necessary 
                                    #Necessary to know the finall cost of the epoch. Item method extracts the numerical value of a PyTorch tensor representing the loss. 

    return total_loss / len(loader.dataset) #Total loss / number of graphs in the loader


def model_pipeline(model, train_loader, val_loader, test_loader):
    criterion = loss_VGAE
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    losses = {"train": [], "val": []}

    for epoch in range(80):
        train_loss = train(epoch, criterion, model, optimizer, train_loader)
        val_loss = validate(criterion, model, val_loader)
        losses["train"].append(train_loss)
        losses["val"].append(val_loss)

        display.clear_output(wait=True)

        plt.plot(losses["train"], label="training loss")
        plt.plot(losses["val"], label="validation loss")

        plt.legend()
        plt.pause(0.000001)
        plt.show()   