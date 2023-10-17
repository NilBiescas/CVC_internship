from pathlib import Path

import os

# ROOT
ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

# PROJECT TREE

CHECKPOINTS = Path('/home/nbiescas/Desktop/CVC/CVC_internship/CheckPoints')
DATA = ROOT / 'datasets'
TRAINING = ROOT / 'training'

SRC = ROOT / 'src'
TRAIN_GRAPH = SRC / 'train_graph.pkl'
VAL_GRAPH   = SRC / 'val_graph.pkl'