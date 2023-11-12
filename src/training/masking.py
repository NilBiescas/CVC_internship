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
from ..models.VGAE import device
from .utils import (get_model, 
                    compute_crossentropy_loss, 
                    get_optimizer, 
                    get_relative_positons, 
                    load_graphs, 
                    weighted_edges, 
                    concat_geom_edge_featurs,
                    region_encoding,
                    spatial_features)

from ..evaluation import (SVM_classifier, 
                          kmeans_classifier, 
                          compute_auc_mc, get_f1, 
                          get_binary_accuracy_and_f1,
                          plot_predictions)

def train_funsd(model, criterion, optimizer, train_graph, config):
    
    model.train()

    x_pred, x_true, n_scores = model(train_graph, train_graph.ndata['Geometric'].to(device), mask_rate = config['mask_rate'])
    n_true = train_graph.ndata['label'].to(device)
    n_scores = n_scores.to(device)

    n_loss = compute_crossentropy_loss(n_scores, n_true)
    #Reconstruction loss
    recons_loss = criterion(x_pred.to(device), x_true.to(device))
    train_loss = recons_loss + n_loss

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    macro, micro, _, _ = get_f1(n_scores, n_true)

    auc = compute_auc_mc(n_scores, n_true)
    #wandb.log({"Train loss": train_loss.item(), "Train node classification loss": n_loss.item(), "Train edge classification loss": e_loss.item()})
    
    return train_loss, macro, auc

def validation_funsd(model, criterion, val_graph):
    model.eval()
    with torch.no_grad():
        #feat = val_graph.ndata['feat'].to(device)

        x_pred_val, x_true_val, n_scores_val = model(val_graph, val_graph.ndata['Geometric'].to(device), mask_rate = 0.0)

        val_n_loss = compute_crossentropy_loss(n_scores_val.to(device), val_graph.ndata['label'].to(device))     
        recons_loss = criterion(x_pred_val.to(device), x_true_val.to(device))

        val_loss = recons_loss + val_n_loss

        macro, micro, precision, recall = get_f1(n_scores_val, val_graph.ndata['label'].to(device))
        wandb.log({"precision macro": precision, "recall macro": recall})
        auc = compute_auc_mc(n_scores_val.to(device), val_graph.ndata['label'].to(device))

        #wandb.log({"Validation loss": val_loss.item(), "Validation node classification loss": val_n_loss.item(), "Validation edge classification loss": val_e_loss.item()})

    return val_loss, macro, auc, precision

def test_funsd(model, test_graph, criterion):
    with torch.no_grad():
        model.eval()

        x_pred_test, x_true_test, n_scores_test = model(test_graph, test_graph.ndata['Geometric'].to(device), mask_rate = 0.0)

        recons_loss = criterion(x_pred_test.to(device), x_true_test)
        n_loss = compute_crossentropy_loss(n_scores_test.to(device), test_graph.ndata['label'].to(device))
        test_loss = recons_loss + n_loss

        auc = compute_auc_mc(n_scores_test.to(device), test_graph.ndata['label'].to(device))
        macro, micro, precision, recall = get_f1(n_scores_test, test_graph.ndata['label'].to(device))
        
        ################* STEP 4: RESULTS ################

        print("\n### BEST RESULTS ###")
        print("Precision nodes macro: {:.4f}".format(precision))
        print("Recall nodes macro: {:.4f}".format(recall))
        print("AUC Nodes: {:.4f}".format(auc))
        print("F1 Nodes: Macro {:.4f} - Micro {:.4f}".format(macro, micro))
        
    return test_loss

def test_evaluation(model, train_graph, criterion, config):
    data_test = FUNSD_loader(train=False) #Loading test set graphs
    data_test.get_info()

    test_graph = dgl.batch(data_test.graphs)

    # Add features
    test_graph.ndata['Geometric'] = add_features(test_graph)

    # Add weighted edges
    test_graph.edata['weights'] = weighted_edges(test_graph)
    test_graph = test_graph.int().to(device)
    
    test_loss = test_funsd(model, test_graph, criterion)
    pred_kmeans = kmeans_classifier(model, train_graph, test_graph, config)
    pred_svm    = SVM_classifier(model, train_graph, test_graph, config)

    start, end = config['images']['start'], config['images']['end']
    plot_predictions(data_test, test_graph, pred_svm, path = config['output_svm'], start = start, end = end)
    plot_predictions(data_test, test_graph, pred_kmeans, path = config['output_kmeans'], start = start, end = end)
    return test_loss.item()

def add_features(graph):
    #features = concat_geom_edge_featurs(graph)
    features = graph.ndata['geom']
    #features = torch.cat((features, concat_geom_edge_featurs(graph)), dim = 1)
    # Adding areas
    area = lambda geom: (geom[:, 2] - geom[:, 0]) * (geom[:, 3] - geom[:, 1])
    features = torch.cat((features, area(graph.ndata['geom']).unsqueeze(dim=1)), dim = 1)
    # Region encoding
    features = torch.cat((features, region_encoding(graph)), dim = 1)
    return features

def masking_funsd(config):
    # Loading data
    data = FUNSD_loader(train=True)
    data.get_info()
    train_graphs, val_graphs, _, _ = train_test_split(data.graphs, torch.ones(len(data.graphs), 1), test_size=0.2, random_state=42)
    print("-> Number of training graphs: ", len(train_graphs))
    print("-> Number of validation graphs: ", len(val_graphs))

    #Graph for training
    train_graph = dgl.batch(train_graphs)    
    #Graph for validating
    val_graph = dgl.batch(val_graphs)

    # Adding features
    train_graph.ndata['Geometric'] = add_features(train_graph)
    val_graph.ndata['Geometric']   = add_features(val_graph)

    # Weighted edges
    train_graph.edata['weights'] = weighted_edges(train_graph)
    val_graph.edata['weights']   = weighted_edges(val_graph)

    train_graph = train_graph.int().to(device)
    val_graph   = val_graph.int().to(device)

    if config['network']['checkpoint'] is None:
        model = get_model(config)
        optimizer = get_optimizer(model, config)

        criterion = torch.nn.MSELoss(reduction=config['reduce'])
        wandb.watch(model)

        total_train_loss = 0
        total_validation_loss = 0
        best_val_precision = 0
        for epoch in range(config['epochs']):

            train_loss, macro, auc = train_funsd(model, criterion, optimizer, train_graph, config)
            val_tot_loss, val_macro, val_auc, precision = validation_funsd(model, criterion, val_graph)

            total_train_loss += train_loss.item()
            total_validation_loss += val_tot_loss.item()

            if precision > best_val_precision:
                best_val_precision = precision
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

    test_loss = test_evaluation(best_model, train_graph, criterion, config)

    print("Train Loss: {:.4f} | Validation Loss: {:.4f} | Test Loss: {:.4f}".format(total_train_loss, total_validation_loss, test_loss))
    return best_model