import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F 
from IPython import display
import matplotlib.pyplot as plt
import wandb
from tqdm.auto import tqdm

#Implementation of the loss: https://github.com/JuliaSun623/VGAE_dgl/blob/main/train.py. Later on implement the norm parameter


# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loss_VGAE(output, adjacency_matrices_target, means_list, log_std_list):
    total_loss = 0
    for pred, adj_matrix, mean, log_std in zip(output, adjacency_matrices_target, means_list, log_std_list):
        loss = F.binary_cross_entropy(pred.view(-1), adj_matrix.view(-1))
        total_loss += loss.item()

        kl_divergence = 0.5 / pred.size(0) * (
                1 + 2 * log_std - mean ** 2 - torch.exp(log_std) ** 2).sum(
            1).mean()
        total_loss -= kl_divergence
        
    return total_loss

@torch.no_grad()  # prevent this function from computing gradients see https://pytorch.org/docs/stable/generated/torch.no_grad.html
def validate(criterion, model, loader):

    val_loss = 0
    correct = 0
    total_nodes = 0

    model.eval()

    for graph, adjacency_matrices_target in loader:
        #sparse_adj = sparse_adj[0]

        graph = graph.to(device)
        features = graph.ndata.pop('feat').to(device)

        output, means_list, log_std_list = model(graph, features)
        loss = criterion(output, adjacency_matrices_target, means_list, log_std_list)
        val_loss += loss.item()
        
        for pred, adj_matrix in zip(output, adjacency_matrices_target):
            A = pred == adj_matrix
            correct += torch.sum(A)/2 + torch.diag(A).sum() * 0.5                                                    
            total_nodes += (graph.num_nodes() ** 2 + graph.num_nodes())/2

    val_loss /= len(loader.dataset)
    accuracy = 100. * correct / total_nodes
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, total_nodes,
        accuracy))
    
    return val_loss

def train(epoch, criterion, model, optimizer, loader):
    
    total_loss = 0.0

    model.train()

    for batch_idx, (graph, adjacency_matrices_target) in enumerate(loader): 

        optimizer.zero_grad()
        #Adjacency_matrices is a list of tensors. I move the tensors to the device in the collate function of the dataloader
        graph = graph.to(device)
        features = graph.ndata.pop('feat').to(device)

        output, means_list, log_std_list = model(graph, features)  #The output is a list containing the different predictions for the graphs
        loss = criterion(output, adjacency_matrices_target, means_list, log_std_list)

        loss.backward()
        optimizer.step()
        
        # print loss every N iterations
        if batch_idx % 50 == 0:
            print('Train Epoch: {} \tGraphs_seen: {}% \tLoss: {:.6f}'.format(
                epoch, round(batch_idx * loader.batch_size / len(loader), 2), loss.item()))


        total_loss += loss.item()  #.item() is very important here? Why? To sum up the indivuals losses of each of the samples seen in an epoch. Necessary 
                                    #Necessary to know the finall cost of the epoch. Item method extracts the numerical value of a PyTorch tensor representing the loss. 

    return total_loss / len(loader.dataset) #Total loss / number of graphs in the loader


def model_pipeline(model, train_loader, val_loader, test_loader):
    wandb.watch(model)
    criterion = loss_VGAE
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)
    losses = {"train": [], "val": []}

    for epoch in tqdm(range(15)):
        train_loss = train(epoch, criterion, model, optimizer, train_loader)
        val_loss = validate(criterion, model, val_loader)
        wandb.log({"train_loss": train_loss,
                   "val_loss": val_loss}, step = epoch)
        
        losses["train"].append(train_loss)
        losses["val"].append(val_loss)

        #display.clear_output(wait=True)
#
        #plt.plot(losses["train"], label="training loss")
        #plt.plot(losses["val"], label="validation loss")
#
        #plt.legend()
        #plt.pause(0.000001)
        #plt.show()   