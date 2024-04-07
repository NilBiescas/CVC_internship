# GeoContrastNet
This repor contains the code for the paper GeoContrastNet. The model is divided in two stages, so you can train or test one of the two stages.
To perform the name entity and link predicition tasks you will need to use stage-2


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
Training 
To train in stage 2 and evaluate on FUNSD you will need to run this command
```
python main.py --run-name (name of you yaml file)
```
Test
To test the model you will need to do:
```
python main.py --run-name (name of you yaml file) --checkpoint (path to saved checkpoints)
```
As you see the only difference for training or testing is specifying the pretrained weights.

The way how it works is that you train the firts stage1 and that training geneartest the graph used in stage2. So you have the graph used in stage1 and then the graph used in stage2 genereated in stage1.
I provided all the necessary graphs and configs to repeat the experimetns and that is not necessary for you to train again to obtain the graphs used in stage2. You can go directly and train them and see the results.
I also provided the pretrain weights of the model leearn in stage1 for each of the different configs and the pretrain weights used in stage2.



To try stage-1 results you will find the graphs necessary in the folder setups_stage1. 


You need to change the HERE variable to your own path in the root.env file
