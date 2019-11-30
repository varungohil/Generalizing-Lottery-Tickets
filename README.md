---   
<div align="center">    
 
# NeurIPS Reproducibility Challenge     
</div>
 
## Description   
This repository contains code to replicate the experiments given in NeurIPS 2019 paper "One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers"


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
```bash
# source folder
cd Generalizing-Lottery-Ticket/src   

# run module (example: mnist as your main contribution)   
python mnist_trainer.py    
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
