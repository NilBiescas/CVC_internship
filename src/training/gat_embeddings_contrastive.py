import torch
import wandb
import sys
import dgl
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import random
import numpy as np
import json

sys.path.append("..") 

from ..models import get_model_2
from ..data.Dataset import (FUNSD_loader, 
                            dataloaders_funsd, 
                            kmeans_graphs, 
                            edgesAggregation_kmeans_graphs)

from ..models.autoencoders import device
from .utils import (get_model, 
                    compute_crossentropy_loss, 
                    get_optimizer,
                    get_scheduler,
                    weighted_edges, 
                    region_encoding)

from ..evaluation import (SVM_classifier, 
                          kmeans_classifier, 
                          compute_auc_mc, 
                          get_f1,
                          conf_matrix,
                          plot_predictions,
                          get_accuracy)

def train_funsd(model, optimizer, train_loader, config):
    model.train()
    nodes_predictions = []
    nodes_ground_truth = []
    total_train_loss = 0
    for train_graph in train_loader:
        
        train_graph = train_graph.to(device)
        feat = train_graph.ndata['feat'].to(device)
        labels = train_graph.ndata['label'].to(device)
        
        x_pred = model(train_graph, feat, mask_rate = config['mask_rate']).to(device)

        train_loss = compute_crossentropy_loss(x_pred, labels)
        #Reconstruction lossç
        wandb.log({'classification loss': train_loss.item()})
        total_train_loss += train_loss
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        nodes_predictions.append(x_pred)
        nodes_ground_truth.append(labels)

    nodes_predictions = torch.cat(nodes_predictions, dim = 0)
    nodes_ground_truth = torch.cat(nodes_ground_truth)

    macro, micro, _, _ = get_f1(nodes_predictions, nodes_ground_truth)
    auc = compute_auc_mc(nodes_predictions, nodes_ground_truth)
    accuracy_train = get_accuracy(nodes_predictions, nodes_ground_truth)

    return total_train_loss, macro, auc, accuracy_train

def validation_funsd(model, val_loader):
    model.eval()
    nodes_predictions = []
    nodes_ground_truth = []
    total_validation_loss = 0
    with torch.no_grad():
        for val_graph in val_loader:
            
            val_graph = val_graph.to(device)
            feat = val_graph.ndata['feat'].to(device)
            x_true = val_graph.ndata['label'].to(device)

            x_pred = model(val_graph, feat, mask_rate = 0).to(device)

            val_n_loss = compute_crossentropy_loss(x_pred, x_true)     
            total_validation_loss += val_n_loss

            nodes_predictions.append(x_pred)
            nodes_ground_truth.append(x_true)

        nodes_predictions = torch.cat(nodes_predictions, dim = 0)
        nodes_ground_truth = torch.cat(nodes_ground_truth)

        
        macro, micro, precision, recall = get_f1(nodes_predictions, nodes_ground_truth)
        wandb.log({"precision macro": precision, "recall macro": recall})
        auc = compute_auc_mc(nodes_predictions, nodes_ground_truth)
        accuracy = get_accuracy(nodes_predictions, nodes_ground_truth)
    return total_validation_loss, macro, auc, precision, accuracy

def test_funsd(model, test_loader, config):
    model.eval()
    nodes_predictions = []
    nodes_ground_truth = []
    total_test_loss = 0

    with torch.no_grad():
        for test_graph in test_loader:
            
            test_graph = test_graph.to(device)
            feat = test_graph.ndata['feat'].to(device)
            x_true = test_graph.ndata['label'].to(device)

            x_pred = model(test_graph, feat, mask_rate = 0).to(device)

            val_n_loss = compute_crossentropy_loss(x_pred, x_true)     
            total_test_loss += val_n_loss

            nodes_predictions.append(x_pred.to('cpu'))
            nodes_ground_truth.append(x_true.to('cpu'))

    nodes_predictions = torch.cat(nodes_predictions, dim = 0)
    nodes_ground_truth = torch.cat(nodes_ground_truth)
    
    macro_f1, micro, precision, recall = get_f1(nodes_predictions, nodes_ground_truth)
    auc = compute_auc_mc(nodes_predictions, nodes_ground_truth)
    accuracy = get_accuracy(nodes_predictions, nodes_ground_truth)
    # Compute confusion matrix

    _, indices = torch.max(nodes_predictions, dim=1)
    indices = indices.cpu().detach().numpy()
    nodes_ground_truth = nodes_ground_truth.cpu().detach().numpy()

    conf_matrix(nodes_ground_truth, indices, config["output_dir"], title="Confusion Matrix - Test Set - MODEL")
    ################* STEP 4: RESULTS ################
    print("\n### BEST RESULTS ###")
    print("Precision nodes macro: {:.4f}".format(precision))
    print("Recall nodes macro: {:.4f}".format(recall))
    print("Accuracy nodes: {:.4f}".format(accuracy))
    print("AUC Nodes: {:.4f}".format(auc))
    print("F1 Nodes: Macro {:.4f} - Micro {:.4f}".format(macro_f1, micro))
    
    wandb.log({"Test precision macro": precision, "Test recall macro": recall, "Test accuracy": accuracy, "Test AUC": auc, "Test F1 macro": macro_f1})
    data = {    
            "accuracy": accuracy,
            "f1": macro_f1,
            "precision": precision,
            "recall": recall
        }
    print("Saving metrics.json")
    with open(config['output_dir'] / 'metrics.json', 'w') as f:
            json.dump(data, f)

    return total_test_loss / len(test_loader.dataset)

def test_evaluation(model, train_loader, test_loader, config):
    data_test = FUNSD_loader(train=False) #Loading test set graphs with Kmeans edges

    test_loss = test_funsd(model, test_loader)
    
    train_graph = dgl.batch(train_loader.dataset)
    test_graph = dgl.batch(test_loader.dataset)
    
    train_graph = train_graph.to(device)
    test_graph = test_graph.to(device)

    pred_kmeans = kmeans_classifier(model, train_graph, test_graph, config)
    pred_svm    = SVM_classifier(model, train_graph, test_graph, config)

    test_graph = dgl.batch(data_test.graphs)
    start, end = config['images']['start'], config['images']['end']
    plot_predictions(data_test, test_graph, pred_svm, path = config['output_svm'], start = start, end = end)
    plot_predictions(data_test, test_graph, pred_kmeans, path = config['output_kmeans'], start = start, end = end)
    return test_loss.item()

def contrastive_training_embeddings(config):
    # Load the learned embeddings

    train_graphs, _ = dgl.load_graphs(config['train_graphs'])
    test_graphs, _ = dgl.load_graphs(config['test_graphs'])
    
    train_graphs, validation_graphs, _, _ = train_test_split(train_graphs, torch.ones(len(train_graphs), 1), test_size=0.1, random_state=42)

    train_loader = torch.utils.data.DataLoader(train_graphs, batch_size=config['batch_size'], collate_fn = dgl.batch, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_graphs, batch_size=config['batch_size'], collate_fn = dgl.batch, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_graphs, batch_size=config['batch_size'], collate_fn = dgl.batch, shuffle=False)

    config['activation'] = F.relu
    try:
        if config['network']['checkpoint'] is None:
            model = get_model_2(config['model_name'], config).to(device)
            optimizer = get_optimizer(model, config)
            scheduler = get_scheduler(optimizer, config)
 
            wandb.watch(model)

            total_train_loss = 0
            total_validation_loss = 0
            best_val_auc = 0
            for epoch in range(config['epochs']):

                train_loss, macro, auc, accuracy_train = train_funsd(model, optimizer, train_loader, config)
                val_tot_loss, val_macro, val_auc, precision, accuracy_val = validation_funsd(model, validation_loader)
                scheduler.step()

                total_train_loss += train_loss.item()
                total_validation_loss += val_tot_loss.item()

                if val_auc > best_val_auc:
                    torch.save(model.state_dict(), config['weights_dir'] / f"epoch_{epoch}.pth")
                    best_val_auc = val_auc
                    best_model = model

                wandb.log({"Train loss": train_loss.item(), 
                           "Train node macro": macro, 
                           "Train node auc": auc,
                           "Validation loss": val_tot_loss.item(), 
                           "Validation node macro": val_macro,
                           "Validation node auc": val_auc,
                           "Validation node accuracy": accuracy_val,
                           "Train node accuracy": accuracy_train})
                
                print("Epoch {:05d} | TrainLoss {:.4f} | TrainF1-MACRO-node {:.4f} | TrainAUC-PR-node {:.4f} | ValLoss {:.4f} | ValF1-MACRO-node {:.4f} | ValAUC-PR-node {:.4f} |"
                        .format(epoch, train_loss.item(), macro, auc, val_tot_loss.item(), val_macro, val_auc))

            total_train_loss /= config['epochs']; total_validation_loss /= config['epochs']
            print("Train Loss: {:.4f} | Validation Loss: {:.4f} | Test Loss: {:.4f}".format(total_train_loss, total_validation_loss), end='')
        else:
            print("Loading model from checkpoint")
            best_model = get_model_2(config['model_name'], config)
            best_model.load_state_dict(torch.load(config['network']['checkpoint']))
            best_model = best_model.to(device)
    except KeyboardInterrupt:
        pass
    #test_loss = test_evaluation(best_model, train_loader, config)
    test_loss = test_funsd(best_model, test_loader, config)

    print(" | Test Loss: {:.4f}".format(test_loss))
    return best_model