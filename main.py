from pathlib import Path
import torch
import wandb
import argparse
import pprint
import numpy as np
import random
import yaml

from src.models.autoencoders import GAE, VGAEModel
from src.training.funsd import _funsd
from src.training.masking import masking_funsd
from src.training.masking_gat import Gat_masking_funsd
from src.training.masking_subgraph import Sub_Graphs_masking
from src.training.edges_masking_subgraph import Sub_Graphs_masking_edges
from src.training import get_training_function

from src.paths import TASKS_YAML
from utils import LoadConfig

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)
np.random.seed(42)
random.seed(42)

if __name__ == '__main__':

    wandb.login()
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--run-name', type=str, default='run32')
    args = parser.parse_args()

    config = LoadConfig(args.run_name)

    print(f"\n{'- '*10}CONFIGURATION{' -'*10}\n")
    pprint.pprint(config, indent=10, width=1)
    print("\n\n")
    with wandb.init(project="masking", config = config):

        wandb.run.name = config['run_name']
        
        training_function = get_training_function(config['train_method'])
        model = training_function(config)

        torch.save(model.state_dict(), config['weights_dir'] / "initial_weights.pth")
        print("Done")




    #
        #if (config['train_method'] == 'FUNSD'):
        #    print(f"\n{'- '*10}FUNSD{' -'*10}\n")
        #    model = _funsd(config)
#
        #if (config['train_method']  == 'SELF_FUNSD'):
        #    print(f"\n{'- '*10}SELF_FUNSD{' -'*10}\n")
        #    model = masking_funsd(config)
#
        #if (config['train_method'] == 'GAT'):
        #    print(f"\n{'- '*10}GAT{' -'*10}\n")
        #    model = Gat_masking_funsd(config)
#
        #if (config['train_method'] == 'Sub_Graphs_masking'):
        #    print(f"\n{'- '*10}Sub_Graphs_masking{' -'*10}\n")
        #    model = Sub_Graphs_masking(config)
        #
        #if (config['train_method'] == 'Sub_Graphs_masking_edges'):
        #    model = Sub_Graphs_masking_edges(config)