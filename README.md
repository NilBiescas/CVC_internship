# GeoContrastNet

This repository provides the implementation of GeoContrastNet, as detailed in our paper. GeoContrastNet is a two-stage Graph Neural Network designed for Named Entity Recognition and Link Prediction in document entities identified by YOLO. The first stage learns geometric representations, which, alongside visual features, are utilized in the second stage for the tasks mentioned.

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

2. **Optional: For Custom Graph Construction**
   ```
   git clone https://github.com/andreagemelli/doc2graph.git
   ```

3. **Create and Activate a Conda Environment**
   ```
   conda create --name doc2graph python=3.9.18
   conda activate doc2graph
   ```

4. **Install Requirements**
   ```
   pip install -r requirements.txt
   ```

### Configuration Files
- YAML configuration files for all experiments are provided. You may use these directly or as templates for custom experiments.
- Match YAML files with their corresponding pretrained weights for experimentation.

### Training and Testing

- **Training on FUNSD (Stage 2)**
  ```
  python main.py --run-name <your_yaml_file>
  ```
- **Testing**
  ```
  python main.py --run-name <your_yaml_file> --checkpoint <path_to_pretrained_weights>
  ```

## Additional Resources
- **Stage 1 Graphs**: Located in `setups_stage1` folder.
- **Environment Setup**: Modify the `HERE` variable in `root.env` to your path.

## Contribution
To contribute to the GeoContrastNet project, please follow the standard GitHub pull request process. Ensure your changes are well-documented and tested.
