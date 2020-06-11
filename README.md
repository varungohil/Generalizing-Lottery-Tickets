---   
<div align="center">    
 
# Generalizing Lottery Tickets   

 [![DOI](https://zenodo.org/badge/224994704.svg)](https://zenodo.org/badge/latestdoi/224994704)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>
 

 
## Description   
This repository contains PyTorch code to replicate the experiments given in NeurIPS 2019 paper 

[___"One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers"___](https://arxiv.org/abs/1906.02773v2)

As finding the _winning lottery tickets_ is computationally expensive, we also open-source winning tickets (pretrained and pruned models) we generated during our experiments. Link : [Winning Tickets](https://drive.google.com/drive/folders/1Nd-J4EwmgWbUARYaqe9iCF6efEFf9S2P?usp=sharing)

## How to Setup    
```bash
# clone project   
git clone https://github.com/varungohil/Generalizing-Lottery-Tickets.git  

# install all dependencies   
cd Generalizing-Lottery-Tickets    
pip3 install -r requirements.txt
```

## How to Run
There are 4 files in ```src``` folder:
- train.py             : Use to train the neural network and find the winning ticket
- test.py              : Use to test the accuracy of the trained model
- iterative_pruning.py : Use to iteratively prune the model.
- utils.py             : Contains helper functions used in scripts mentioned above

To support more datasets and architectures, we need to add necessary code to utils.py

### Using train.py
##### Mandatory arguments:
- --architecture : To specify the neural network architecture (vgg19 and resnet50)
- --dataset      : The dataset to train on (cifar10, cifar100, fashionmnist, svhn, cifar10a, cifar10b)
##### Optional arguments:
- --batch-size : To set the batch size while training
- --optimizer  : The optimizer to use for training (sgd and adam). sgd used by default
- --seed : To set the ranodm seed
- --model-saving-path : Path to directory where trained model is saved.

The trained model will be saved for first 5 epochs. For VGG19 it will be saved for every 16<sup>th</sup> epoch. FOr Resnet50, the model will be saved for every 9<sup>th</sup> epoch. For our experiments, while pruning, we reinitialize te model with weights after epoch 2 (late resetting of 1).
```bash
# source folder
cd Generalizing-Lottery-Ticket/src   

# run train.py
python3 train.py --architecture=resnet50 --dataset=cifar10    
```

### Using iterative_pruning.py
##### Mandatory arguments:
- --architecture : To specify the neural network architecture (vgg19 and resnet50)
- --target-dataset      : The dataset to train on (cifar10, cifar100, fashionmnist, svhn, cifar10a, cifar10b)
- --source-dataset      : The dataset using which winning ticket initialization was found (cifar10, cifar100, fashionmnist, svhn, cifar10a, cifar10b)
- --init-path   : Path to model with winning ticket initialization

##### Optional arguments:
- --batch-size : To set the batch size while training
- --optimizer  : The optimizer to use for training (sgd and adam). sgd used by default
- --seed : To set the ranodm seed
- --model-saving-path : Path to directory where trained model is saved.

The script will run 30 pruning iterations which will prune away 99.9% of the weights. The trained and pruned model will be saved at end of each pruning iteration

```bash
# source folder
cd Generalizing-Lottery-Ticket/src   

# run iterative_pruning.py
python3 iterative_pruning.py --architecture=resnet50 --source-dataset=cifar10 --target-dataset=cifar100 --model-saving-path=<path-to-dir-where-models-are-to-be-stored>
```

### Using test.py
##### Mandatory arguments:
- --architecture : To specify the neural network architecture (vgg19 and resnet50)
- --dataset      : The dataset to train on (cifar10, cifar100, fashionmnist, svhn, cifar10a, cifar10b)
- --model-path   : The path to moedl whose accuracy needs to be evaluated.

##### Optional arguments:
- --batch-size : To set the batch size while training

Running this script will print the _Fraction of pruned weights_ in the model and the _Test Accuracy_. 
```bash
# source folder
cd Generalizing-Lottery-Ticket/src   

# run train.py
python3 test.py --architecture=resnet50 --dataset=cifar10 --model-path=<path-to-model>   
```


### Results   
The results of the replicated experiments can be found in plots folder.
  

### Citation 
If you use this repository, kindly cite the reproducibility report and the original paper. The bibtex is given below.
```
@article{Gohil:2020,
  author = {Gohil, Varun and Narayanan, S. Deepak and Jain, Atishay},
  title = {{[Re] One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers}},
  journal = {ReScience C},
  year = {2020},
  month = may,
  volume = {6},
  number = {2},
  pages = {{#4}},
  doi = {10.5281/zenodo.3818619},
  url = {https://zenodo.org/record/3818619/files/article.pdf},
  code_url = {https://github.com/varungohil/Generalizing-Lottery-Tickets},
  code_doi = {10.5281/zenodo.3700320},
  code_swh = {swh:1:dir:8a9e53bc8a9028428bbad6a4e77ae3fedae49d30},
  data_url = {},
  data_doi = {},
  review_url = {https://openreview.net/forum?id=SklFHaqG6S},
  type = {Replication},
  language = {Python},
  domain = {NeurIPS 2019 Reproducibility Challenge},
  keywords = {lottery ticket hypothesis, pytorch}
}

@incollection{NIPS2019_8739,
title = {One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers},
author = {Morcos, Ari and Yu, Haonan and Paganini, Michela and Tian, Yuandong},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {4932--4942},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/8739-one-ticket-to-win-them-all-generalizing-lottery-ticket-initializations-across-datasets-and-optimizers.pdf}
}

```  
### Core Contributors
[Varun Gohil](https://varungohil.github.io), [S. Deepak Narayanan](https://sdeepaknarayanan.github.io), [Atishay Jain](https://github.com/AtishayJain-ML)

### Development
We have a new branch ```dev``` in which pull requests are welcome. We will merge them after reviewing. 

### Contributors
[fcorencot](https://github.com/fcorencoret)
