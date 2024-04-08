# GeoContrastNet

This repository provides the implementation of GeoContrastNet, as detailed in our paper: GeoContrastNet: Contrastive Key-Value Edge Learning for Language-Agnostic Document Understanding. GeoContrastNet is a two-stage Graph Neural Network designed for Named Entity Recognition and Link Prediction in document entities identified by YOLO or any other object detector. The first stage learns geometric representations, which, alongside visual features, are utilized in the second stage for the tasks mentioned.

## Features
- Code for training and testing both stages of the model.
- Pretrained weights for immediate model evaluation.
- Graphs for both stages, including how to generate your own.
- Instructions for replicating paper experiments.

## Getting Started

### Prerequisites
- Git
- Conda

### Setup Instructions

1. **Clone the Repository**
   ```
   git clone https://github.com/NilBiescas/CVC_internship.git
   ```

2. **Optional: For Custom Graph Construction. Not necessary if you want to use the grahps that we provide.**
   ```
   git clone https://github.com/andreagemelli/doc2graph.git
   ```

4. **Create and Activate a Conda Environment**
   ```
   conda create --name doc2graph python=3.9.18
   conda activate doc2graph
   ```

5. **Install Requirements**
   ```
   pip install -r requirements.txt
   ```
6. Enter in paths.py and modify

### Configuration Files
- YAML configuration files for all experiments are provided. You may use these directly or as templates for custom experiments.
- Match YAML files with their corresponding pretrained weights for experimentation. Each yaml file goes with their matching pretrained weights, they share the same name.

### Training and Testing
## Stage 1
- **Training**
  ```
  python build_graphs.py --run-name <your_yaml_file>
  ```
- **Testing**
  ```
  python build_graphs.py --run-name <your_yaml_file> --checkpoint <path_to_pretrained_weights>
  ```
## Stage 2
- **Training on FUNSD**
  ```
  python main.py --run-name <your_yaml_file>
  ```
- **Testing**
  ```
  python main.py --run-name <your_yaml_file> --checkpoint <path_to_pretrained_weights>
  ```

## Additional Resources
- **Stage 1 YAML files**: Located in `setups_stage1` folder.
- **Stage 2 YAML files**: Located in `setups_stage2` folder.
- **Stage 1 Graphs**: Located in `graphs_stage1` folder.
- **Stage 2 Graphs**: Located in `graphs_stage2` folder.

## Contribution
To contribute to the GeoContrastNet project, please follow the standard GitHub pull request process. Ensure your changes are well-documented and tested.
