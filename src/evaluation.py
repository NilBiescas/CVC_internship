import torch
import wandb
import pandas as pd
import seaborn as sn
from typing import Tuple
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, confusion_matrix)
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

from .models.VGAE import device


def compute_auc_mc(scores, labels):
    scores = scores.detach().cpu().numpy()
    labels = F.one_hot(labels).cpu().numpy()
    # return roc_auc_score(labels, scores)
    return average_precision_score(labels, scores)

def get_binary_accuracy_and_f1(classes, labels : torch.Tensor, per_class = False) -> Tuple[float, list]:

    correct = torch.sum(classes.flatten() == labels)
    accuracy = correct.item() * 1.0 / len(labels)
    classes = classes.detach().cpu().numpy()
    labels = labels.cpu().numpy()

    if not per_class:
        f1 = f1_score(labels, classes, average='macro'), f1_score(labels, classes, average='micro')
    else:
        f1 = precision_recall_fscore_support(labels, classes, average=None)[2].tolist()
    
    return accuracy, f1

def get_f1(logits : torch.Tensor, labels : torch.Tensor, per_class = False) -> tuple:
    """Returns Macro and Micro F1-score for given logits / labels.

    Args:
        logits (torch.Tensor): model prediction logits
        labels (torch.Tensor): target labels

    Returns:
        tuple: macro-f1 and micro-f1
    """
    _, indices = torch.max(logits, dim=1)
    indices = indices.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    if not per_class:
        return f1_score(labels, indices, average='macro'), f1_score(labels, indices, average='micro')
    else:
        return precision_recall_fscore_support(labels, indices, average=None)[2].tolist()

def metrics_MSE_mean(train_loss, graph):
    out_dimensions = graph.ndata['feat'].shape[1]
    node_reconstruction_loss = train_loss * out_dimensions
    graph_reconstructin_loss = (node_reconstruction_loss * graph.num_nodes()) / graph.batch_size

    return node_reconstruction_loss, graph_reconstructin_loss

# In order to comapre the reduction method used in MSELoss
def metrics_MSE_sum(loss, graph):
    out_dimensions = graph.ndata['feat'].shape[1]

    graph_reconstructin_loss = loss / graph.batch_size
    node_reconstruction_loss = loss / graph.num_nodes()
    feature_reconstruction_loss = node_reconstruction_loss / out_dimensions

    return node_reconstruction_loss, graph_reconstructin_loss, feature_reconstruction_loss


def extract_embeddings(graph, model):
    with torch.no_grad():
        model.eval()
        h = graph.ndata['feat'].to(device)
        for layer in model.encoder:
            h = layer(graph, h)

        embeddings = h.cpu().detach().numpy()
        labels = graph.ndata['label'].cpu().detach().numpy()
        return embeddings, labels


def conf_marix(y_true, y_pred):

    data = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(data, columns=np.array(['answer', 'header', 'other', 'question']), index = np.array(['answer', 'header', 'other', 'question']))

    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    
    plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4) #for label size
    sn.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='d') # font size
    plt.show()

def kmeans_classifier(model, train_graph, test_graph):
    from sklearn.cluster import KMeans
    with torch.no_grad():

        embeddings_train, labels_train = extract_embeddings(train_graph, model)
        embeddings_test, labels_test   = extract_embeddings(test_graph, model)

        kmeans = KMeans(n_clusters=4, random_state=0).fit(embeddings_train)
        groups = kmeans.labels_

        group1_idx = np.where(groups == 0)[0]
        group2_idx = np.where(groups == 1)[0]
        group3_idx = np.where(groups == 2)[0]
        group4_idx = np.where(groups == 3)[0]

        group1_labels, group2_labels, group3_labels, group4_labels = labels_train[group1_idx], labels_train[group2_idx], labels_train[group3_idx], labels_train[group4_idx]

        group1_labels, group2_labels, group3_labels, group4_labels = np.bincount(group1_labels), np.bincount(group2_labels), np.bincount(group3_labels), np.bincount(group4_labels)

        if not all(map(lambda a: len(a) != 0, [group1_labels, group2_labels, group3_labels, group4_labels])):
            return
        
        mapping = {'0': np.argmax(group1_labels), '1': np.argmax(group2_labels), '2': np.argmax(group3_labels), '3': np.argmax(group4_labels)}
        print(mapping)

        pred = kmeans.predict(embeddings_test)
        pred = [mapping[str(i)] for i in pred]

        accuracy = accuracy_score(labels_test, pred)
        f1 = f1_score(labels_test, pred, average='macro')
        precision = precision_score(labels_test, pred, average='macro')
        recall = recall_score(labels_test, pred, average='macro')
        
        accuracy *= 100; f1 *= 100; precision *= 100; recall *= 100 #To percentage
        accuracy, f1, precision, recall = int(accuracy), int(f1), int(precision), int(recall) #To int
        print("Accuracy kmeans: {:.4f} | F1 Score kmeans: {:.4f} | Precision kmeans: {:.4f} | Recall kmeans: {:.4f}".format(accuracy, f1, precision, recall))
        wandb.log({"Accuracy kmeans": accuracy, "F1 Score kmeans": f1, "Precision kmeans": precision, "Recall kmeans": recall})
        conf_marix(labels_test, pred)



def SVM_classifier(model, train_graph, test_graph):
    from sklearn.svm import SVC

    with torch.no_grad():

        embeddings_train, labels_train = extract_embeddings(train_graph, model)
        embeddings_test, labels_test   = extract_embeddings(test_graph, model)

        clf = SVC(kernel='rbf', C=1, gamma='auto')
        clf.fit(embeddings_train, labels_train)
        pred = clf.predict(embeddings_test)
        accuracy = accuracy_score(labels_test, pred)
        f1 = f1_score(labels_test, pred, average='macro')
        precision = precision_score(labels_test, pred, average='macro')
        recall = recall_score(labels_test, pred, average='macro')

        # Percentatge
        accuracy *= 100; f1 *= 100; precision *= 100; recall *= 100
        accuracy, f1, precision, recall = int(accuracy), int(f1), int(precision), int(recall)
        print("Accuracy SVM: {:.4f} | F1 Score SVM: {:.4f} | Precision SVM: {:.4f} | Recall SVM: {:.4f}".format(accuracy, f1, precision, recall))
        wandb.log({"Accuracy SVM": accuracy, "F1 Score SVM": f1, "Precision SVM": precision, "Recall SVM": recall})
        conf_marix(labels_test, pred)