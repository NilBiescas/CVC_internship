import torch
import wandb
import sys
import dgl
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import random
import numpy as np

sys.path.append("..") 
from ..data.Data_Loaders import FUNSD_loader
from ..data.Dataset import dataloaders_funsd, kmeans_graphs

from ..models.VGAE import device
from .utils import (get_model, 
                    compute_crossentropy_loss, 
                    get_optimizer,
                    get_scheduler,
                    weighted_edges, 
                    region_encoding)

from ..evaluation import (SVM_classifier, 
                          kmeans_classifier, 
                          compute_auc_mc, get_f1,
                          plot_predictions)

def train_funsd(model, criterion, optimizer, train_loader, config):
    model.train()
    nodes_predictions = []
    nodes_ground_truth = []
    total_train_loss = 0
    for train_graph in train_loader:
        
        train_graph = train_graph.to(device)

        x_pred, x_true, n_scores = model(train_graph, train_graph.ndata['Geometric'].to(device), mask_rate = config['mask_rate'])
        n_true = train_graph.ndata['label'].to(device)
        n_scores = n_scores.to(device)

        n_loss = compute_crossentropy_loss(n_scores, n_true)
        #Reconstruction loss
        recons_loss = criterion(x_pred.to(device), x_true.to(device))
        train_loss = recons_loss + n_loss
        total_train_loss += train_loss

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        nodes_predictions.append(n_scores)
        nodes_ground_truth.append(n_true)

    nodes_predictions = torch.cat(nodes_predictions, dim = 0)
    nodes_ground_truth = torch.cat(nodes_ground_truth)

    macro, micro, _, _ = get_f1(nodes_predictions, nodes_ground_truth)
    auc = compute_auc_mc(nodes_predictions, nodes_ground_truth)

    return total_train_loss, macro, auc

def validation_funsd(model, criterion, val_loader):
    model.eval()
    nodes_predictions = []
    nodes_ground_truth = []
    total_validation_loss = 0
    with torch.no_grad():
        for val_graph in val_loader:
            
            val_graph = val_graph.to(device)

            x_pred_val, x_true_val, n_scores_val = model(val_graph, val_graph.ndata['Geometric'].to(device), mask_rate = 0.0)

            val_n_loss = compute_crossentropy_loss(n_scores_val.to(device), val_graph.ndata['label'].to(device))     
            recons_loss = criterion(x_pred_val.to(device), x_true_val.to(device))
            val_loss = recons_loss + val_n_loss
            total_validation_loss += val_loss

            nodes_predictions.append(n_scores_val)
            nodes_ground_truth.append(val_graph.ndata['label'])
        
        nodes_predictions = torch.cat(nodes_predictions, dim = 0)
        nodes_ground_truth = torch.cat(nodes_ground_truth)

        
        macro, micro, precision, recall = get_f1(nodes_predictions, nodes_ground_truth)
        wandb.log({"precision macro": precision, "recall macro": recall})
        auc = compute_auc_mc(nodes_predictions, nodes_ground_truth)

    return total_validation_loss, macro, auc, precision

def test_funsd(model, test_loader, criterion):
    model.eval()
    with torch.no_grad():
        nodes_predictions = []
        nodes_ground_truth = []
        total_test_loss = 0
        for test_graph in test_loader:
            
            test_graph = test_graph.to(device)

            x_pred_test, x_true_test, n_scores_test = model(test_graph, test_graph.ndata['Geometric'].to(device), mask_rate = 0.0)

            recons_loss = criterion(x_pred_test.to(device), x_true_test)
            n_loss = compute_crossentropy_loss(n_scores_test.to(device), test_graph.ndata['label'].to(device))
            test_loss = recons_loss + n_loss
            total_test_loss += test_loss

            nodes_predictions.append(n_scores_test)
            nodes_ground_truth.append(test_graph.ndata['label'])

        nodes_predictions = torch.cat(nodes_predictions, dim = 0)
        nodes_ground_truth = torch.cat(nodes_ground_truth)

        auc = compute_auc_mc(nodes_predictions, nodes_ground_truth)
        macro, micro, precision, recall = get_f1(nodes_predictions, nodes_ground_truth)
        
        ################* STEP 4: RESULTS ################

        print("\n### BEST RESULTS ###")
        print("Precision nodes macro: {:.4f}".format(precision))
        print("Recall nodes macro: {:.4f}".format(recall))
        print("AUC Nodes: {:.4f}".format(auc))
        print("F1 Nodes: Macro {:.4f} - Micro {:.4f}".format(macro, micro))
    
    return total_test_loss / len(test_loader.dataset)

def test_evaluation(model, train_loader, criterion, config):
    data_test = FUNSD_loader(train=False) #Loading test set graphs

    if config['kmeans_partition']:
        test_loader = kmeans_graphs(train = False, config = config)
    else:
        print("-> Loading random partitioned graphs")
        test_loader = dataloaders_funsd(train = False, config = config)

    test_loss = test_funsd(model, test_loader, criterion)
    
    train_graph = dgl.batch([train_graph for train_graph in train_loader])
    test_graph = dgl.batch([test_graph for test_graph in test_loader])
    
    train_graph = train_graph.to(device)
    test_graph = test_graph.to(device)

    pred_kmeans = kmeans_classifier(model, train_graph, test_graph, config)
    pred_svm    = SVM_classifier(model, train_graph, test_graph, config)

    test_graph = dgl.batch(data_test.graphs)
    start, end = config['images']['start'], config['images']['end']
    plot_predictions(data_test, test_graph, pred_svm, path = config['output_svm'], start = start, end = end)
    plot_predictions(data_test, test_graph, pred_kmeans, path = config['output_kmeans'], start = start, end = end)
    return test_loss.item()

def add_features(graph):
    features = graph.ndata['geom']
    # Adding areas
    area = lambda geom: (geom[:, 2] - geom[:, 0]) * (geom[:, 3] - geom[:, 1])
    features = torch.cat((features, area(graph.ndata['geom']).unsqueeze(dim=1)), dim = 1)
    # Region encoding
    features = torch.cat((features, region_encoding(graph)), dim = 1)
    return features

def Sub_Graphs_masking(config):
    # Loading data
    if config['kmeans_partition']:
        train_loader, val_loader = kmeans_graphs(train = True, config = config)
    else:
        print("-> Loading random partitioned graphs")
        train_loader, val_loader = dataloaders_funsd(train = True, config = config)

    try:
        if config['network']['checkpoint'] is None:
            model = get_model(config)
            optimizer = get_optimizer(model, config)
            scheduler = get_scheduler(optimizer, config)

            criterion = torch.nn.MSELoss(reduction=config['reduce'])
             
            wandb.watch(model)

            total_train_loss = 0
            total_validation_loss = 0
            best_val_auc = 0
            for epoch in range(config['epochs']):

                train_loss, macro, auc = train_funsd(model, criterion, optimizer, train_loader, config)
                val_tot_loss, val_macro, val_auc, precision = validation_funsd(model, criterion, val_loader)
                scheduler.step()

                total_train_loss += train_loss.item()
                total_validation_loss += val_tot_loss.item()

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model = model

                wandb.log({"Train loss": train_loss.item(), 
                           "Train node macro": macro, 
                           "Train node auc": auc,
                           "Validation loss": val_tot_loss.item(), 
                           "Validation node macro": val_macro,
                           "Validation node auc": val_auc})
                
                print("Epoch {:05d} | TrainLoss {:.4f} | TrainF1-MACRO-node {:.4f} | TrainAUC-PR-node {:.4f} | ValLoss {:.4f} | ValF1-MACRO-node {:.4f} | ValAUC-PR-node {:.4f} |"
                        .format(epoch, train_loss.item(), macro, auc, val_tot_loss.item(), val_macro, val_auc))

            total_train_loss /= config['epochs']; total_validation_loss /= config['epochs']
        else:
            best_model = get_model(config)
            best_model.load_state_dict(torch.load(config['network']['checkpoint']))
            best_model = best_model.to(device)
    except KeyboardInterrupt:
        pass
    test_loss = test_evaluation(best_model, train_loader, criterion, config)

    print("Train Loss: {:.4f} | Validation Loss: {:.4f} | Test Loss: {:.4f}".format(total_train_loss, total_validation_loss, test_loss))
    return best_model