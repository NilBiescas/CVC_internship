#from ..data.Data_Loaders import FUNSD_loader
import sys
sys.path.append("..") 
from ..data.Data_Loaders import FUNSD_loader
from sklearn.model_selection import train_test_split
import dgl

from ..models.VGAE import device
from ..models.VGAE import GAE, GSage_AE, GIN_AE, GAT_AE
import torch
import wandb


def metrics_MSE_mean(train_loss, graph):
    out_dimensions = graph.ndata['feat'].shape[1]
    node_reconstruction_loss = train_loss * out_dimensions
    graph_reconstructin_loss = (node_reconstruction_loss * graph.num_nodes()) / graph.batch_size

    return node_reconstruction_loss, graph_reconstructin_loss, 

# In order to comapre the reduction method used in MSELoss
def metrics_MSE_sum(loss, graph):
    out_dimensions = graph.ndata['feat'].shape[1]

    graph_reconstructin_loss = loss / graph.batch_size
    node_reconstruction_loss = loss / graph.num_nodes()
    feature_reconstruction_loss = node_reconstruction_loss / out_dimensions

    return node_reconstruction_loss, graph_reconstructin_loss, feature_reconstruction_loss

def validation_funsd(epoch, criterion, model, val_graph, config):
    model.eval()
    with torch.no_grad():
        pred = model(val_graph, val_graph.ndata['feat'].to(device))
        val_loss = criterion(pred, val_graph.ndata['feat'].to(device))
        val_loss = val_loss.item()

        if (epoch % 50) == 0:
            if config.reduce == "mean":
                node_reconstruction_loss, graph_reconstructin_loss = metrics_MSE_mean(val_loss, val_graph) # Mean loss per feature

            elif config.reduce == "sum":
                node_reconstruction_loss, graph_reconstructin_loss, val_loss = metrics_MSE_sum(val_loss, val_graph)
            print(f"\n{'- '*10}Validation{' -'*10}\n")
            wandb.log({"epoch": epoch, "val loss": val_loss, "Val Node reconstruction loss": node_reconstruction_loss, "Val graph reconstructin loss": graph_reconstructin_loss})
            print('\nepoch: {}\tFeature reconstruction loss validation: {:.6f}'.format(epoch, val_loss))
            print('Node reconstruction loss validation:  {}\tGraph reconstruction loss validation: {}'.format(node_reconstruction_loss, graph_reconstructin_loss))

    return val_loss

def train_funsd(epoch, criterion, model, optimizer, graph, config):
    
    model.train()
    
    feat = graph.ndata['feat'].to(device)
    pred = model(graph, feat)

    train_loss = criterion(pred.to(device), feat)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    train_loss = train_loss.item()
    if (epoch % 50) == 0:
        if config.reduce == "mean":
            node_reconstruction_loss, graph_reconstructin_loss = metrics_MSE_mean(train_loss, graph) # Mean loss per feature

        elif config.reduce == "sum":
            node_reconstruction_loss, graph_reconstructin_loss, train_loss = metrics_MSE_sum(train_loss, graph)
        print(f"\n{'- '*10}Training{' -'*10}\n")
        wandb.log({"epoch": epoch, "train loss": train_loss, "Train node reconstruction loss": node_reconstruction_loss, "Train graph reconstructin loss": graph_reconstructin_loss})
        print('\nepoch: {}\tFeature reconstruction loss training: {:.6f}'.format(epoch, train_loss))
        print('Node reconstruction loss training:  {}\tGraph reconstruction loss training: {}'.format(node_reconstruction_loss, graph_reconstructin_loss))

    return train_loss

def test_funsd(model, criterion, config):
    model.eval()
    with torch.no_grad():
        data_test = FUNSD_loader(train=False) #Loading test set graphs

        test_graph = dgl.batch(data_test.graphs)
        test_graph = test_graph.int().to(device)

        features = test_graph.ndata['feat'].to(device)
        pred = model(test_graph, features)
        test_loss = criterion(pred, features)

        test_loss = test_loss.item()
        if config.reduce == "mean":
            node_reconstruction_loss, graph_reconstructin_loss = metrics_MSE_mean(test_loss, test_graph) # Mean loss per feature

        elif config.reduce == "sum":
            node_reconstruction_loss, graph_reconstructin_loss, test_loss = metrics_MSE_sum(test_loss, test_graph)

        print(f"\n{'- '*10}TEST{' -'*10}\n")
        wandb.log({"test loss": test_loss, "Test node reconstruction loss": node_reconstruction_loss, "Test graph reconstructin loss": graph_reconstructin_loss})
        print('\nFeature reconstruction loss test: {:.6f}'.format(test_loss))
        print('Node reconstruction loss test:  {}\tGraph reconstruction loss test: {}'.format(node_reconstruction_loss, graph_reconstructin_loss))


    return test_loss

def _funsd(config):

    data = FUNSD_loader(train=False)
    train_graphs, val_graphs, _, _ = train_test_split(data.graphs, torch.ones(len(data.graphs), 1), test_size=0.20)

    #Graph for training
    train_graph = dgl.batch(train_graphs)
    train_graph = train_graph.int().to(device)

    #Graph for testing
    val_graph = dgl.batch(val_graphs)
    val_graph = val_graph.int().to(device)

    #Dimensions of the autencoder
    layers_dimensions = config.layers_dimensions # Tuple = (100, 25, 10)

    if config.model == 'SAGE':
        model = GSage_AE(layers_dimensions).to(device)
    elif config.model == 'GAE':
        model = GAE(layers_dimensions).to(device)
    elif config.model == 'GIN':
        model = GIN_AE(layers_dimensions).to(device)
    elif config.model == 'GAT':
        model = GAT_AE(layers_dimensions).to(device)
    
    #torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss(reduction=config.reduce)
    wandb.watch(model)

    total_train_loss = 0
    total_validation_loss = 0
    for epoch in range(config.epochs):
        train_loss = train_funsd(epoch, criterion, model, optimizer, train_graph, config)
        validation_loss = validation_funsd(epoch, criterion, model, val_graph, config)

        #scheduler.step(validation_loss)
        total_train_loss += train_loss
        total_validation_loss += validation_loss

    #total_train_loss /= train_graph.batch_size
    #total_validation_loss /= val_graph.batch_size
    test_loss = test_funsd(model, criterion, config)
    print("Average train loss: {}\nAverage validation loss: {}\nTest loss: {}".format(total_train_loss, total_validation_loss, test_loss))
    wandb.log({"test loss": test_loss})
    return model
