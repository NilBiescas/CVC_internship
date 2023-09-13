from Data_Loaders import loaders
from VGAE import VGAE, VGAEModel
from pathlib import Path
import torch
from training import model_pipeline
import wandb

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


BASE_DIR = Path("/home/nbiescas/Desktop/CVC/CVC_internship") #or Path().absolute()
DATA_PATH = BASE_DIR / "omniglot.npz"
from training import device

if __name__ == '__main__':
    wandb.login()
    train_loader, val_loader, test_loader = loaders(DATA_PATH)
    
    model = VGAEModel(2, 10, 15).double()
    if torch.cuda.is_available():
        model = model.to(device)

    print("Starting Training")
    with wandb.init(project="VGAE"):
        wandb.run.name = 'VAGE RUN-1'
        model_pipeline(model, train_loader, val_loader, test_loader)
    
    print("Done")