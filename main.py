from pathlib import Path
import torch
import wandb
import argparse
import pprint
import numpy as np

# OWN MODULES
from src.training import get_orchestration_func
from utils import LoadConfig

# Ensure deterministic behavior
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
np.random.seed(42)

if __name__ == '__main__':

    wandb.login()
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--run-name', type=str, default='run144')
    args = parser.parse_args()

    config = LoadConfig(args.run_name)

    print(f"\n{'- '*10}CONFIGURATION{' -'*10}\n")
    pprint.pprint(config, indent=10, width=1)
    print("\n\n")
    with wandb.init(project="experiments_finals", config = config, name = config['run_name']):
        
        orchestration_func = get_orchestration_func(config['train_method']) # Load orchestration function
        model = orchestration_func(config)

        
        print("Ja s'ha acabat")
