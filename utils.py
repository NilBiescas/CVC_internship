import yaml
import os
from pathlib import Path

def createDir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def LoadConfig(test_name):
    with open(Path("/home/nbiescas/Desktop/CVC/CVC_internship/Setups") / (test_name + ".yaml")) as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    
    ROOT = Path('/home/nbiescas/Desktop/CVC/CVC_internship/runs') / test_name
    opt['run_name'] = test_name

    opt["root_dir"]         = ROOT
    opt["weights_dir"]      = ROOT / "weights"
    opt["output_dir"]       = ROOT / "images"
    opt['output_svm']       = ROOT / 'images' / "svm"
    opt['output_kmeans']    = ROOT / 'images' / "kmeans"
    opt['json_kmeans']      = ROOT / 'images' / "kmeans"
    opt['json_svm']         = ROOT / 'images' / "svm"
    
    createDir(opt["root_dir"])
    createDir(opt["weights_dir"])
    createDir(opt["output_dir"])
    createDir(opt["output_svm"])
    createDir(opt["output_kmeans"])

    opt["network"]["checkpoint"] = opt["network"].get("checkpoint", None)

    return opt
