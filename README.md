# GeoContrastNet
This repoitory contains the code for the paper GeoContrastNet. The model explained in the paper, consists of a two stage graph neural network that classifies the entites of a document obtained using YOLO. In the firts stage we learn geometric representations that are then used in the second stage along with visual features to do name entity recognition and link prediction. The code to train and test both stages is available. 

Stage-1 receives as input a graph and stores a new graph used for stage-2. We provided all the components in order to:
- Obtain a config file and pretrained weights and try the model.
- Replicate the experiments done in the paper.
Then this resources include: Both the graphs used in stage-1 and the ones obtained to use in stage-2, also the pretrained weights learned in stage-1 and for stage-2. 

# Setup

Clonse this repository
```
git clone https://github.com/NilBiescas/CVC_internship.git
```
You will need the graphs to try the experiments, we already provide them, but if you want to construc it yourself you will need to clonse this repository
```
git clone https://github.com/andreagemelli/doc2graph.git
```
Once you have cloned the repository, create a conda environment.
```
conda create doc2graph python=3.9.18
```
and then intall the requirments.txt
```
pip install -r requirments.txt
```
# YAML files
To try the models you will need to use a yaml file. We provide the yaml files used for all the experiments explaind in the paper. You can use those or if you want to try your experiments you can use them as template. 
Each yaml file goes with the pretrained weights obtained. If you want to try one of the experiments you pick the yaml file and weights file wiht the same name.

Training 
To train in stage 2 and evaluate on FUNSD you will need to run this command
```
python main.py --run-name (name of you yaml file)
```
Test
To test the model you will need to do:
```
python main.py --run-name (file1.yaml) --checkpoint (path to file1.pth)
```
As you see the only difference for training or testing is specifying the pretrained weights.

I provided all the necessary graphs and configs to repeat the experimetns and that is not necessary for you to train again to obtain the graphs used in stage2. You can go directly and train them and see the results.
I also provided the pretrain weights of the model leearn in stage1 for each of the different configs and the pretrain weights used in stage2.



To try stage-1 results you will find the graphs necessary in the folder setups_stage1. 


You need to change the HERE variable to your own path in the root.env file
