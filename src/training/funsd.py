import torch
import wandb
import sys
import dgl
from sklearn.model_selection import train_test_split

sys.path.append("..") 

from ..data.Data_Loaders import FUNSD_loader
from ..models.VGAE import device
from .utils import get_model, compute_crossentropy_loss, log_wandb
from ..evaluation import SVM_classifier, metrics_MSE_mean, metrics_MSE_sum, kmeans_classifier

def validation_funsd(model, criterion, val_graph, epoch, config):
    model.eval()
    with torch.no_grad():
        feat = val_graph.ndata['feat'].to(device)

        n_scores, e_scores, pred = model(val_graph, feat)
        recons_loss = criterion(pred.to(device), feat)

        #n_loss = compute_crossentropy_loss(n_scores.to(device), val_graph.ndata['label'].to(device))
        #e_loss = compute_crossentropy_loss(e_scores.to(device), val_graph.edata['label'].to(device))
        #val_loss = n_loss + e_loss + recons_loss

        # Reconstruction loss for the autencoder
        recons_loss = recons_loss.item()

        if (epoch % 50) == 0:
            if config.reduce == "mean":
                node_reconstruction_loss, graph_reconstructin_loss = metrics_MSE_mean(recons_loss, val_graph) # Mean loss per feature

            elif config.reduce == "sum":
                node_reconstruction_loss, graph_reconstructin_loss, recons_loss = metrics_MSE_sum(recons_loss, val_graph)
            
            log_wandb("Validation", recons_loss, node_reconstruction_loss, graph_reconstructin_loss)

    return recons_loss

def train_funsd(model, criterion, optimizer, train_graph, epoch, config):
    
    model.train()
    
    feat = train_graph.ndata['feat'].to(device)

    n_scores, e_scores, pred = model(train_graph, feat)
    recons_loss = criterion(pred.to(device), feat)

    n_loss = compute_crossentropy_loss(n_scores.to(device), train_graph.ndata['label'].to(device))
    e_loss = compute_crossentropy_loss(e_scores.to(device), train_graph.edata['label'].to(device))
    train_loss = n_loss + e_loss + recons_loss

    optimizer.zero_grad()
    train_loss.backward()
    n = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step()
    
    recons_loss = recons_loss.item()
    if (epoch % 50) == 0:
        if config.reduce == "mean":
            node_reconstruction_loss, graph_reconstructin_loss = metrics_MSE_mean(recons_loss, train_graph) # Mean loss per feature

        elif config.reduce == "sum":
            node_reconstruction_loss, graph_reconstructin_loss, recons_loss = metrics_MSE_sum(recons_loss, train_graph)
        
        log_wandb("Training", recons_loss, node_reconstruction_loss, graph_reconstructin_loss)

    return recons_loss

def test_funsd(model, test_graph, criterion, config):
    with torch.no_grad():
        model.eval()
        feat = test_graph.ndata['feat'].to(device)
        n_scores, e_scores, pred = model(test_graph, feat)
        recons_loss = criterion(pred.to(device), feat)
        
        n_loss = compute_crossentropy_loss(n_scores.to(device), test_graph.ndata['label'].to(device))
        e_loss = compute_crossentropy_loss(e_scores.to(device), test_graph.edata['label'].to(device))
        test_loss = n_loss + e_loss + recons_loss

        print(test_loss)
        recons_loss = recons_loss.item()
        if config.reduce == "mean":
            node_reconstruction_loss, graph_reconstructin_loss = metrics_MSE_mean(recons_loss, test_graph) # Mean loss per feature

        elif config.reduce == "sum":
            node_reconstruction_loss, graph_reconstructin_loss, recons_loss = metrics_MSE_sum(recons_loss, test_graph)

        log_wandb("Test", recons_loss, node_reconstruction_loss, graph_reconstructin_loss)

    return recons_loss

def test_evaluation(model, train_graph, criterion, config):
    data_test = FUNSD_loader(train=False) #Loading test set graphs
    test_graph = dgl.batch(data_test.graphs)
    test_graph = test_graph.int().to(device)
    
    test_loss = test_funsd(model, test_graph, criterion, config)
    kmeans_classifier(model, train_graph, test_graph)
    SVM_classifier(model, train_graph, test_graph)

    return test_loss


def _funsd(config):

    data = FUNSD_loader(train=True)
    train_graphs, val_graphs, _, _ = train_test_split(data.graphs, torch.ones(len(data.graphs), 1), test_size=0.2, random_state=42)
    print("Number of training graphs: ", len(train_graphs))
    print("Number of validation graphs: ", len(val_graphs))
    #Graph for training
    train_graph = dgl.batch(train_graphs)
    train_graph = train_graph.int().to(device)

    #Graph for validating
    val_graph = dgl.batch(val_graphs)
    val_graph = val_graph.int().to(device)

    # Selecting model
    model = get_model(config, data)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss(reduction=config.reduce)
    wandb.watch(model)

    total_train_loss = 0
    total_validation_loss = 0
    for epoch in range(config.epochs):

        train_loss = train_funsd(model, criterion, optimizer, train_graph, epoch, config)
        validation_loss = validation_funsd(model, criterion, val_graph, epoch, config)

        total_train_loss += train_loss
        total_validation_loss += validation_loss

    total_train_loss /= config.epochs; total_validation_loss /= config.epochs
    test_loss = test_evaluation(model, train_graph, criterion, config)

    print("Train Loss: {:.4f} | Validation Loss: {:.4f} | Test Loss: {:.4f}".format(total_train_loss, total_validation_loss, test_loss))
    return model