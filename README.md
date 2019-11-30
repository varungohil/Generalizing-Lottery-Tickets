---   
<div align="center">    
 
# NeurIPS Reproducibility Challenge     
</div>
 
## Description   
This repository contains code to replicate the experiments given in NeurIPS 2019 paper 

___"One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers"___


## How to Setup    
```bash
# clone project   
git clone https://github.com/varungohil/Generalizing-Lottery-Tickets.git  

# install all dependencies   
cd Generalizing-Lottery-Ticket    
pip3 install requirements.txt
```

## How to Run
There are 4 file in ```src``` folder:
- train.py             : Use to train the neural network and find the winning ticket
- test.py              : Use to test the accuracy of the trained model
- iterative_pruning.py : Use to iteratively prune the model.
- utils.py             : Contains helper functions used in scripts mentioned above

### Using train.py
Mandatory arguments:
- --architecture : To specify the neural network architecture (vgg19 and resnet50)
- --dataset      : The dataset to train on (cifar10, cifar100, fashionmnist, svhn)
Optional arguments:
- --batch-size : To set the batch size while training
- --optimizer  : The optimizer to use for training (sgd and adam). sgd used by default
- --seed : To set the ranodm seed
- --model-saving-path : Path to directory where trained model is saved.

The trained model will be saved for first 5 epochs. For VGG19 it will be saved for every 16<sup>th</sup> epoch. FOr Resnet50, the model will be saved for every 9<sup>th</sup> epoch. For our experiments, while pruning, we reinitialize te model with weights after epoch 2 (late resetting of 1).
```bash
# source folder
cd Generalizing-Lottery-Ticket/src   

# run train,py
python3 train.py --architecture=resnet50 --dataset=cifar10    
```

## Main Contribution   
TODO
  

### Citation   
```
@article{Morcos2019OneTT,
  title={One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers},
  author={Ari S. Morcos and Haonan Yu and Michela Paganini and Yuandong Tian},
  journal={ArXiv},
  year={2019},
  volume={abs/1906.02773}
}
```   
