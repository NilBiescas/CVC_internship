from Data.d_preprocess import load_npz_file, read_letter
from models.VGAE import VGAE
from torch.utils.data import DataLoader
import dgl
from pathlib import Path

BASE_DIR = Path("/home/nbiescas/Desktop/CVC/CVC_internship") #or Path().absolute()
DATA_PATH = BASE_DIR / "omniglot.npz"

def collate(graphs):
    batched_graph = dgl.batch(graphs)
    return batched_graph

if __name__ == '__main__':
    trainset, validset, testset = load_npz_file(DATA_PATH)

    Train_Graphs = [read_letter(graph) for graph in trainset]
    Valid_Graphs = [read_letter(graph) for graph in validset]
    Test_Graphs  = [read_letter(graph) for graph in testset]

    train_loader = DataLoader(Train_Graphs, batch_size=32, shuffle=True,
                         collate_fn=collate)
    valid_loader = DataLoader(Valid_Graphs, batch_size=32, collate_fn=collate)
    test_loader  = DataLoader(Test_Graphs, batch_size=32, collate_fn=collate)

    model = VGAE(2, 10)