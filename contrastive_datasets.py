import torch
import pickle
import dgl
import os
import matplotlib.pyplot as plt
from functools import partial
import argparse
import torch.optim as optim

from src.models import get_model_2
from pytorch_metric_learning import losses, miners, distances, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from src.training.utils import get_model, get_activation, get_scheduler
from src.training.utils_contrastive import obtain_embeddings, create_plots
from src.models.contrastive_models import Simple_edge_encoder
from src.data.Dataset import kmeans_graphs, Dataset_Kmeans_Graphs
from utils import LoadConfig, createDir
from src.data.utils import concat_paragraph2graph_edges


def train(model, loss_func, mining_func, train_loader, optimizer, epoch):    
    model.train()
    total_loss = 0
    
    for batch_idx, (graph, labels) in enumerate(train_loader):

        graph = graph.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        embeddings = model(graph)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        #loss = loss_func(embeddings, labels)

        total_loss += loss.item()
        loss.backward()

        optimizer.step()
        if batch_idx % 20 == 0:
            #print("Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(epoch, batch_idx, loss, mining_func.num_triplets))
            print("Epoch {} Iteration {}: Loss = {}".format(epoch, batch_idx, loss))

    return total_loss / (batch_idx + 1)

def get_all_embeddings(dataset, model):
    tester = testers.BaseTester(dataloader_num_workers=2)
    return tester.get_all_embeddings(dataset, model, collate_fn=collate)

def test(train_set, test_set, model, accuracy_calculator):
    model.eval()
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)

    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(query = train_embeddings, reference = test_embeddings, query_labels = train_labels, reference_labels = test_labels)
    
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))
    return accuracies["precision_at_1"]

def collate(batch):
    graphs, labels = map(list, zip(*batch))
    graphs = dgl.batch(graphs)

    return graphs, torch.cat(labels, dim=0)

def vector_func(config, edges):
    node_feat_list = []
    for node_feat in config['features']['node']:
        node_feat = edges.src[node_feat]
        if len(node_feat.size()) == 1:
            node_feat = node_feat.unsqueeze(1)
        node_feat_list.append(node_feat)

    for edge_feat in config['features']['edge']:
        edge_feat = edges.data[edge_feat]
        if len(edge_feat.size()) == 1:
            edge_feat = edge_feat.unsqueeze(1)
        node_feat_list.append(edge_feat)
    
    msg = torch.cat(node_feat_list, dim=1)
    return {'m': msg}


def message_of_the_edges(config: dict, g : dgl.graph):
    message_edge = []
    for edge_feat in config['features']['edge']:
        edge_feat = g.edata[edge_feat]
        if len(edge_feat.size()) == 1:
            edge_feat = edge_feat.unsqueeze(1)
        message_edge.append(edge_feat)

    msg = torch.cat(message_edge, dim=1)
    g.edata['m'] = msg
    return g

# Update features of the nodes
def contrastive_features(loader: list, model, config, path_dir):
    createDir(config['root_dir'] / 'graphs_contrastive')
    with torch.no_grad():
        model.eval()
        graphs = []
        for graph, _ in loader:
            graph = graph.to(device)

            # Obtain the new embeddings
            embeddings = model(graph)

            graph.ndata['feat'] = embeddings
            graphs.extend(dgl.unbatch(graph))

    dgl.save_graphs(str(config['root_dir'] / 'graphs_contrastive' / path_dir), graphs)

"""
RECORDA QUE HAS CAMBIAT UNA PART DEL CODI FONT DE LA LLIBRERIA DELS TRIPLETS A LA FUNCIO compute_all_embeddings
xmin, ymin, xmax, ymax = box
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--run-name', type=str, default='run55')
    args = parser.parse_args()

    config = LoadConfig(args.run_name)

    device = 'cuda' if torch.cuda.is_available() else "cpu"

    #print("Distance metric {}".format(config['contrastive_learning']['distance_metric']))
    
    #distance = getattr(distances, config['contrastive_learning']['distance_metric'])()
    
    mining_func = miners.TripletMarginMiner(margin = config['contrastive_learning']['margin'], 
                                            #distance=distance, 
                                            type_of_triplets=config['contrastive_learning']['type_of_triplets'])
    
    loss_func = losses.TripletMarginLoss(margin=config['contrastive_learning']['margin'], 
                                         #distance = distance, 
                                         triplets_per_anchor=config['contrastive_learning']['type_of_triplets'])
    
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k = 1)


    vector_func_partial = partial(vector_func, config)
    with open(config['pickle_path_train_kmeans'], 'rb') as kmeans_train, open(config['pickle_path_test_kmeans'], 'rb')  as kmeans_test:

        train_graphs = pickle.load(kmeans_train)
        test_graphs = pickle.load(kmeans_test)

    batch = dgl.batch(train_graphs)
    batch.apply_edges(vector_func_partial)
    train_graphs = dgl.unbatch(batch)
    
    batch = dgl.batch(test_graphs)
    batch.apply_edges(vector_func_partial)
    test_graphs = dgl.unbatch(batch)
    
    # Datasets
    train_dataset = Dataset_Kmeans_Graphs(train_graphs) # Train
    test_dataset = Dataset_Kmeans_Graphs(test_graphs)   # Test

    # Loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn = collate, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], collate_fn = collate, shuffle=False)
    
    config['activation'] = get_activation(config['activation'])
    #model = get_model_2(config['model_name'], config).to(device)
    model = get_model(config)
    #model = Simple_edge_encoder(config['layers_dimensions'])

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    num_epochs = config['epochs']
    loss_evolution = []
    precision_evolution = []

    # Scheduler
    scheduler = get_scheduler(optimizer, config)
    max_precision = 0
    
    if config.get('pretrainedWeights', False) == False:
        for epoch in range(1, num_epochs + 1):
            loss = train(model, loss_func, mining_func, train_loader, optimizer, epoch)
            loss_evolution.append(loss)
            precision = test(train_dataset, test_dataset, model, accuracy_calculator)
            precision_evolution.append(precision)
            if precision > max_precision:
                print("Saving model")
                max_precision = precision
                torch.save(model, config["weights_dir"] / f'model_{epoch}.pth')
            scheduler.step()
            
        plt.plot(loss_evolution)
        plt.savefig(config['root_dir'] / 'loss.png')
        plt.close()
        plt.plot(precision_evolution)
        plt.savefig(config['root_dir'] / 'precision.png')
        plt.close()
    
    # Load best model
    model = torch.load(os.path.join(config["weights_dir"], os.listdir(config["weights_dir"])[-1]))
    model.eval()
    # T-SNE Visualization
    createDir(config['root_dir'] / 'train_and_test_embeddings')
    createDir(config['root_dir'] / 'train_embeddings')
    createDir(config['root_dir'] / 'test_embeddings')

    print("Obtaining embeddings for train and test datasets")
    embeddings, labels = obtain_embeddings(train_loader, test_loader, model,train = True, test = True)
    create_plots(embeddings, labels, dir_path = config['root_dir'] / 'train_and_test_embeddings', config=config)

    print("Obtaining embeddings for train dataset")
    embeddings_train, labels_train = obtain_embeddings(train_loader, test_loader, model, train = True, test = False)
    create_plots(embeddings_train, labels_train, dir_path = config['root_dir'] / 'train_embeddings', config = config)

    print("Obtaining embeddings for test dataset")
    embeddings_test, labels_test = obtain_embeddings(train_loader, test_loader, model, train = False, test = True)
    create_plots(embeddings_test, labels_test, dir_path = config['root_dir'] / 'test_embeddings', config = config)
    
    contrastive_features(train_loader, model, config, 'train_contrastive.bin')
    contrastive_features(test_loader, model, config, 'test_contrastive.bin')
