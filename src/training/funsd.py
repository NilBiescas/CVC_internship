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
from .utils import get_model, compute_crossentropy_loss, log_wandb
from ..evaluation import SVM_classifier, metrics_MSE_mean, metrics_MSE_sum, kmeans_classifier, compute_auc_mc, get_f1, get_binary_accuracy_and_f1

#torch.backends.cudnn.deterministic = True
#torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
#torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)
#np.random.seed(42)
#random.seed(42)

def validation_funsd(model, criterion, val_graph, epoch, config):
    model.eval()
    with torch.no_grad():
        feat = val_graph.ndata['feat'].to(device)

        val_n_scores, val_e_scores, pred_features = model(val_graph, feat)
        recons_loss = criterion(pred_features.to(device), feat)

        val_n_loss = compute_crossentropy_loss(val_n_scores.to(device), val_graph.ndata['label'].to(device))
        val_e_loss = compute_crossentropy_loss(val_e_scores.to(device), val_graph.edata['label'].to(device))
        val_loss = val_n_loss + val_e_loss + recons_loss

        macro, micro = get_f1(val_n_scores, val_graph.ndata['label'].to(device))
        auc = compute_auc_mc(val_e_scores.to(device), val_graph.edata['label'].to(device))

        wandb.log({"Validation loss": val_loss.item(), "Validation node classification loss": val_n_loss.item(), "Validation edge classification loss": val_e_loss.item()})
        #log_wandb("Validation", recons_loss, node_reconstruction_loss, graph_reconstructin_loss)

    return val_loss, macro, auc

def train_funsd(model, criterion, optimizer, train_graph, epoch, config):
    
    model.train()
    
    feat = train_graph.ndata['feat'].to(device)

    n_scores, e_scores, pred_features = model(train_graph, feat)
    recons_loss = criterion(pred_features.to(device), feat)

    n_loss = compute_crossentropy_loss(n_scores.to(device), train_graph.ndata['label'].to(device))
    e_loss = compute_crossentropy_loss(e_scores.to(device), train_graph.edata['label'].to(device))
    train_loss = n_loss + e_loss + recons_loss

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    macro, micro = get_f1(n_scores, train_graph.ndata['label'].to(device))
    auc = compute_auc_mc(e_scores.to(device), train_graph.edata['label'].to(device))

    wandb.log({"Train loss": train_loss.item(), "Train node classification loss": n_loss.item(), "Train edge classification loss": e_loss.item()})
    #log_wandb("Training", recons_loss, node_reconstruction_loss, graph_reconstructin_loss)

    return train_loss, macro, auc

def test_funsd(model, test_graph, criterion, config):
    with torch.no_grad():
        model.eval()
        n_scores, e_scores, pred_features = model(test_graph, test_graph.ndata['feat'].to(device))

        recons_loss = criterion(pred_features.to(device), test_graph.ndata['feat'].to(device))
        n_loss = compute_crossentropy_loss(n_scores.to(device), test_graph.ndata['label'].to(device))
        e_loss = compute_crossentropy_loss(e_scores.to(device), test_graph.edata['label'].to(device))
        test_loss = n_loss + e_loss + recons_loss

        auc = compute_auc_mc(e_scores.to(device), test_graph.edata['label'].to(device))
        _, preds = torch.max(F.softmax(e_scores, dim=1), dim=1)

        accuracy, f1 = get_binary_accuracy_and_f1(preds, test_graph.edata['label'])
        _, classes_f1 = get_binary_accuracy_and_f1(preds, test_graph.edata['label'], per_class=True)

        macro, micro = get_f1(n_scores, test_graph.ndata['label'].to(device))

        ################* STEP 4: RESULTS ################
        print("\n### BEST RESULTS ###")
        print("AUC {:.4f}".format(auc))
        print("Accuracy {:.4f}".format(accuracy))
        print("F1 Edges: Macro {:.4f} - Micro {:.4f}".format(f1[0], f1[1]))
        print("F1 Edges: None {:.4f} - Pairs {:.4f}".format(classes_f1[0], classes_f1[1]))
        print("F1 Nodes: Macro {:.4f} - Micro {:.4f}".format(macro, micro))

        #log_wandb("Test", recons_loss, node_reconstruction_loss, graph_reconstructin_loss)

    return test_loss

def test_evaluation(model, train_graph, criterion, config):
    data_test = FUNSD_loader(train=False) #Loading test set graphs
    data_test.get_info()

    test_graph = dgl.batch(data_test.graphs)
    test_graph = test_graph.int().to(device)
    
    test_loss = test_funsd(model, test_graph, criterion, config)
    kmeans_classifier(model, train_graph, test_graph, config)
    SVM_classifier(model, train_graph, test_graph, config)

    return test_loss.item()

def _funsd(config):

    data = FUNSD_loader(train=True)
    data.get_info()

    train_graphs, val_graphs, _, _ = train_test_split(data.graphs, torch.ones(len(data.graphs), 1), test_size=0.2, random_state=42)
    print("-> Number of training graphs: ", len(train_graphs))
    print("-> Number of validation graphs: ", len(val_graphs))

    #Graph for training
    train_graph = dgl.batch(train_graphs)
    train_graph = train_graph.int().to(device)

    #Graph for validating
    val_graph = dgl.batch(val_graphs)
    val_graph = val_graph.int().to(device)

    # Selecting model
    model = get_model(config, data)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = torch.nn.MSELoss(reduction=config.reduce)
    wandb.watch(model)

    total_train_loss = 0
    total_validation_loss = 0
    best_val_auc = 0
    for epoch in range(config.epochs):

        train_loss, macro, auc = train_funsd(model, criterion, optimizer, train_graph, epoch, config)
        val_tot_loss, val_macro, val_auc = validation_funsd(model, criterion, val_graph, epoch, config)

        total_train_loss += train_loss.item()
        total_validation_loss += val_tot_loss.item()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = model

        print("Epoch {:05d} | TrainLoss {:.4f} | TrainF1-MACRO {:.4f} | TrainAUC-PR {:.4f} | ValLoss {:.4f} | ValF1-MACRO {:.4f} | ValAUC-PR {:.4f} |"
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