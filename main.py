from Data_Loaders import loaders
from VGAE import VGAE, VGAEModel
from pathlib import Path
import torch
from training import model_pipeline


BASE_DIR = Path("/home/nbiescas/Desktop/CVC/CVC_internship") #or Path().absolute()
DATA_PATH = BASE_DIR / "omniglot.npz"
from training import device

if __name__ == '__main__':

    train_loader, val_loader, test_loader = loaders(DATA_PATH)
   
    model = VGAEModel(2, 10, 15)
    if torch.cuda.is_available():
        model = model.to(device)

    print("Starting Training")
    model_pipeline(model, train_loader, val_loader, test_loader)