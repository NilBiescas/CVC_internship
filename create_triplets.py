from itertools import product
import argparse
import pickle 
import torch
import dgl

def mine_groups(group0, group1, type_class):
    # Using list comprehension for efficiency
    file_store = 'mining_triplets.txt'
    # Open file in append mode if it does not exist create it
    with open(file_store, 'a+') as file:
        file.write(type_class + '\n')
        for anchor, positive in product(group0, group0):
            if anchor != positive:
                write_triplets = ''
                for negative in group1:
                    write_triplets += str(anchor.item()) + ' ' + str(positive.item()) + ' ' + str(negative.item()) + '\n'
                file.write(write_triplets)
                    

if __name__ == '__main__':
    train_graph = '/home/nbiescas/Desktop/CVC/CVC_internship/PKL_Graphs/kmeans_contrastive/train_kmeans_contrastive.pkl'
    
    
    with open(train_graph, 'rb') as pkl_train:
        train_graphs = pickle.load(pkl_train)
    batch = dgl.batch(train_graphs)

    group0 = torch.where(batch.ndata['label'] == 0)[0]
    group1 = torch.where(batch.ndata['label'] == 1)[0]
    group2 = torch.where(batch.ndata['label'] == 2)[0]
    group3 = torch.where(batch.ndata['label'] == 3)[0]

    mine_groups(group0, group1, type_class='0-1')
    mine_groups(group0, group2, type_class='0-2')
    mine_groups(group0, group3, type_class='0-3')
    mine_groups(group1, group2, type_class='1-2')
    mine_groups(group1, group3, type_class='1-3')
    mine_groups(group2, group3, type_class='2-3')

