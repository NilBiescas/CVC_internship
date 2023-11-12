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
from .utils import get_model, compute_crossentropy_loss, get_optimizer, get_relative_positons, load_graphs
from ..evaluation import SVM_classifier, kmeans_classifier, compute_auc_mc, get_f1, get_binary_accuracy_and_f1, plot_predictions

def validation_funsd(model, criterion, val_graph, config):
    model.eval()
    with torch.no_grad():
        feat = val_graph.ndata['feat'].to(device)
        val_loss = 0
        macro, auc = 0, 0

        #feat = val_graph.ndata['geom'].to(device)
        val_n_scores, val_e_scores, pred_features, val_bbox_pred, val_discrete_pos_pred = model(val_graph, feat)

        # Bounding Box loss
        if config.bounding_box_classification:
            bbox_loss = criterion(val_bbox_pred.to(device), val_graph.ndata['geom'].to(device))
            val_loss += bbox_loss

        if config.relative_position_classification:
            discrete_loss = compute_crossentropy_loss(val_discrete_pos_pred.to(device), val_graph.edata['discrete_info'].to(device))
            val_loss += discrete_loss

        if config.node_classification:
            val_n_loss = compute_crossentropy_loss(val_n_scores.to(device), val_graph.ndata['label'].to(device))
            val_loss += val_n_loss
            
        if config.edge_classification:
            val_e_loss = compute_crossentropy_loss(val_e_scores.to(device), val_graph.edata['label'].to(device))
            val_loss += val_e_loss
        
        recons_loss = criterion(pred_features.to(device), feat)
        if config.node_reconstruction:
            macro, micro = get_f1(val_n_scores, val_graph.ndata['label'].to(device))
        
        if config.edge_classification:
            auc = compute_auc_mc(val_e_scores.to(device), val_graph.edata['label'].to(device))
        
        if config.relative_position_classification:
            auc = compute_auc_mc(val_discrete_pos_pred.to(device), val_graph.edata['discrete_info'].to(device))
            wandb.log({"Validation AUC discrete": auc})
            
        #wandb.log({"Validation loss": val_loss.item(), "Validation node classification loss": val_n_loss.item(), "Validation edge classification loss": val_e_loss.item()})

    return val_loss, macro, auc

def train_funsd(model, criterion, optimizer, train_graph, config):
    
    model.train()
    macro, auc = 0, 0
    train_loss = 0
    feat = train_graph.ndata['feat'].to(device)
    #feat = train_graph.ndata['geom']
    n_scores, e_scores, pred_features, bbox_pred, discrete_pos = model(train_graph, feat)

    # Bounding Box loss
    if config.bounding_box_classification:
        bbox_loss = criterion(bbox_pred.to(device), train_graph.ndata['geom'].to(device))
        train_loss += bbox_loss
    # Relative position loss
    if config.relative_position_classification:
        discrete_loss = compute_crossentropy_loss(discrete_pos.to(device), train_graph.edata['discrete_info'].to(device))
        train_loss += discrete_loss
    # Node classification
    if config.node_classification:
        n_loss = compute_crossentropy_loss(n_scores.to(device), train_graph.ndata['label'].to(device))
        train_loss += n_loss
    # Edge classification
    if config.edge_classification:
        e_loss = compute_crossentropy_loss(e_scores.to(device), train_graph.edata['label'].to(device))
        train_loss += e_loss

    #Reconstruction loss
    recons_loss = criterion(pred_features.to(device), feat)
    train_loss += recons_loss

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    if config.node_classification:
        macro, micro = get_f1(n_scores, train_graph.ndata['label'].to(device))

    if config.edge_classification:
        auc = compute_auc_mc(e_scores.to(device), train_graph.edata['label'].to(device))

    #wandb.log({"Train loss": train_loss.item(), "Train node classification loss": n_loss.item(), "Train edge classification loss": e_loss.item()})

    return train_loss, macro, auc

def test_funsd(model, test_graph, criterion, config):
    with torch.no_grad():
        model.eval()
        feat_test = test_graph.ndata['feat'].to(device)
        #feat_test = test_graph.ndata['geom'].to(device)
        n_scores, e_scores, pred_features, bbox_pred, discrete_pos = model(test_graph, feat_test)

        # Bounding Box loss
        test_loss = 0
        if config.bounding_box_classification:
            bbox_loss = criterion(bbox_pred.to(device), test_graph.ndata['geom'].to(device))
            test_loss += bbox_loss

        # Relative position loss
        if config.relative_position_classification:
            discrete_loss = compute_crossentropy_loss(discrete_pos.to(device), test_graph.edata['discrete_info'].to(device))
            test_loss += discrete_loss
        
        # Reconstruction loss
        if config.node_reconstruction:
            recons_loss = criterion(pred_features.to(device), feat_test)
            test_loss += recons_loss

        # Node classificiation loss
        if config.node_classification:
            n_loss = compute_crossentropy_loss(n_scores.to(device), test_graph.ndata['label'].to(device))
            test_loss += n_loss
        
        # Edge classification loss
        if config.edge_classification:
            e_loss = compute_crossentropy_loss(e_scores.to(device), test_graph.edata['label'].to(device))
            test_loss += e_loss

        if config.edge_classification:
            auc = compute_auc_mc(e_scores.to(device), test_graph.edata['label'].to(device))
            _, preds = torch.max(F.softmax(e_scores, dim=1), dim=1)

            accuracy, f1  = get_binary_accuracy_and_f1(preds, test_graph.edata['label'])
            _, classes_f1 = get_binary_accuracy_and_f1(preds, test_graph.edata['label'], per_class=True)

        if config.node_classification:
            macro, micro = get_f1(n_scores, test_graph.ndata['label'].to(device))
        
        ################* STEP 4: RESULTS ################
        #print("\n### BEST RESULTS ###")
        #print("AUC Edges: {:.4f}".format(auc))
        #print("Accuracy Edges: {:.4f}".format(accuracy))
        #print("F1 Edges: Macro {:.4f} - Micro {:.4f}".format(f1[0], f1[1]))
        #print("F1 Edges: None {:.4f} - Pairs {:.4f}".format(classes_f1[0], classes_f1[1]))
        #print("F1 Nodes: Macro {:.4f} - Micro {:.4f}".format(macro, micro))
        
        if config.relative_position_classification:
            auc_discrete = compute_auc_mc(discrete_pos.to(device), test_graph.edata['discrete_info'].to(device))
            macro, micro = get_f1(discrete_pos, test_graph.edata['discrete_info'].to(device))
            print("AUC Discrete positions: {:.4f}".format(auc_discrete))
            print("F1 Discrete positions: Macro {:.4f} - Micro {:.4f}".format(macro, micro))

    return test_loss

def test_evaluation(model, train_graph, criterion, config):
    data_test = FUNSD_loader(train=False) #Loading test set graphs
    data_test.get_info()

    if config.relative_position_classification:
        data_test = get_relative_positons(data_test)

    test_graph = dgl.batch(data_test.graphs)
    test_graph = test_graph.int().to(device)
    
    test_loss = test_funsd(model, test_graph, criterion, config)
    pred_kmeans = kmeans_classifier(model, train_graph, test_graph, config)
    pred_svm = SVM_classifier(model, train_graph, test_graph, config)

    if pred_kmeans is not None:
        plot_predictions(data_test, test_graph, pred_kmeans, title = 'Kmeans_predictions', num_graphs = 10)
    plot_predictions(data_test, test_graph, pred_svm, title = 'SVM_predictions', num_graphs = 10)
    return test_loss.item()

def add_features(graphs, config):
    geom_polar = []
    for i in range(graphs.num_nodes()):
        edges = graphs.out_edges(i)
        edge_features = graphs.edata['feat'][edges[1].long()]
        geom_polar.append(torch.cat((graphs.ndata['geom'][i], edge_features.sum(dim=0))))

    geom_polar = torch.stack(geom_polar)
    graphs.ndata['geom_polar'] = geom_polar

def _funsd(config):
    # Loading data
    train_graph, val_graph = load_graphs(load=False)

    train_graph = train_graph.int().to(device)
    val_graph   = val_graph.int().to(device)

    print("-> Number of training graphs: ", len(train_graph.batch_num_nodes()))
    print("-> Number of validation graphs: ", len(val_graph.batch_num_nodes()))

    # Selecting model
    model = get_model(config)
    # Selecting optimizer
    optimizer = get_optimizer(model, config)

    criterion = torch.nn.MSELoss(reduction=config['reduce'])
    wandb.watch(model)

    total_train_loss = 0
    total_validation_loss = 0
    best_val_auc = 0
    for epoch in range(config['epochs']):

        train_loss, macro, auc = train_funsd(model, criterion, optimizer, train_graph, config)
        val_tot_loss, val_macro, val_auc = validation_funsd(model, criterion, val_graph)

        total_train_loss += train_loss.item()
        total_validation_loss += val_tot_loss.item()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = model

        print("Epoch {:05d} | TrainLoss {:.4f} | TrainF1-MACRO-node {:.4f} | TrainAUC-PR-edge {:.4f} | ValLoss {:.4f} | ValF1-MACRO-node {:.4f} | ValAUC-PR-edge {:.4f} |"
                .format(epoch, train_loss.item(), macro, auc, val_tot_loss.item(), val_macro, val_auc))
        wandb.log({"Train F1-MACRO": macro, "Train AUC-PR": auc, "Validation F1-MACRO": val_macro, "Validation AUC-PR": val_auc})

        # Step the scheduler
        if epoch == 700:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 100

    total_train_loss /= config.epochs; total_validation_loss /= config.epochs
    test_loss = test_evaluation(best_model, train_graph, criterion, config)

    print("Train Loss: {:.4f} | Validation Loss: {:.4f} | Test Loss: {:.4f}".format(total_train_loss, total_validation_loss, test_loss))
    return best_model