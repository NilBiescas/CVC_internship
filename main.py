from pathlib import Path
import torch
import wandb
import argparse
import pprint
import numpy as np
import random

from src.models.VGAE import GAE, VGAEModel
from src.training.funsd import _funsd

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)
np.random.seed(42)
random.seed(42)

BASE_DIR = Path("/home/nbiescas/Desktop/CVC/CVC_internship")

if __name__ == '__main__':

    wandb.login()
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument("--reduce", type=str, default="mean")
    parser.add_argument("--model", type=str, default="GAE")
    parser.add_argument("--src-data", type=str, default='FUNSD',
                        help="which data source to use. It can be FUNSD, PAU or CUSTOM")
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--checkpoint-name", type=str, default='default.pth')
    parser.add_argument("--epochs", type=int, default = 200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--layers-dimensions", type=str, default=(1756, 1000, 600))
    args = parser.parse_args()

    cfg = dict(
        reduce              = args.reduce,
        model               = args.model,
        dataset             = args.src_data,
        checkpoint         = args.checkpoint_name,
        epochs              = args.epochs,
        learning_rate       = args.lr,
        layers_dimensions   = eval(args.layers_dimensions),
        input_size          = 1756,
        dropout             = 0.2,
    )

    print(f"\n{'- '*10}CONFIGURATION{' -'*10}\n")
    pprint.pprint(cfg, indent=10, width=1)
    print("\n\n")
    with wandb.init(project="second-version-autencoder", config = cfg):

        config = wandb.config
        wandb.run.name = f"{config.checkpoint}_{config.dataset}_{config.model}_{config.learning_rate}_{config.epochs}_{config.layers_dimensions}_{config.reduce}_{config.dropout}"

        if (args.src_data == 'FUNSD'):
            print(f"\n{'- '*10}FUNSD{' -'*10}\n")
            model = _funsd(config)


        torch.save(model.state_dict(),f'/home/nbiescas/Desktop/CVC/CVC_internship/CheckPoints/{config.checkpoint}')
        print("Done")