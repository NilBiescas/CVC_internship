import torch
from .models.VGAE import device

from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, confusion_matrix)
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

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


        mapping = {'0': np.argmax(group1_labels), '1': np.argmax(group2_labels), '2': np.argmax(group3_labels), '3': np.argmax(group4_labels)}
        print(mapping)

        pred = kmeans.predict(embeddings_test)
        pred = [mapping[str(i)] for i in pred]

        accuracy = accuracy_score(labels_test, pred)
        f1 = f1_score(labels_test, pred, average='macro')
        precision = precision_score(labels_test, pred, average='macro')
        recall = recall_score(labels_test, pred, average='macro')

        conf_marix(labels_test, pred)
        return accuracy, f1, precision, recall



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

        conf_marix(labels_test, pred)
        return accuracy, f1, precision, recall