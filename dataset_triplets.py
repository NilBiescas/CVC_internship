from torch.utils.data import Dataset
from dataclasses import dataclass

@dataclass
class BaseGraph():
    nodes: int = 4
    edges: int = 6
    node_features:int = 5
    edge_features: int = 7
    node_labels: int = 3


l = BaseGraph()

print(l.nodes)

l.nodes = "a"

print(l.nodes)