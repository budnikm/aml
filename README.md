## Asymmetric metric learning

This is the official code that enables the reproduction of the results from our paper:

**Asymmetric metric learning for knowledge transfer**,
Budnik M., Avrithis Y. 
[[arXiv](https://arxiv.org/abs/2006.16331)]

### Content

This repository provides the means to train and test all the models presented in the paper. This includes:

1. Code to train the models with and without the teacher (asymmetric and symmetric).
1. Code to do symmetric and asymmetric testing on rOxford and rParis datasets.
1. Best pre-trainend models (including whitening).

### Dependencies

1. Python3 (tested on version 3.6)
1. Numpy 1.19
1. PyTorch (tested on version 1.4.0)
1. Datasets and base models will be downloaded automatically.


### Training and testing the networks

To train a model use the following script:
```bash
python main.py [-h] [--training-dataset DATASET] [--directory EXPORT_DIR] [--no-val]
                  [--test-datasets DATASETS] [--test-whiten DATASET]
                  [--val-freq N] [--save-freq N] [--arch ARCH] [--pool POOL]
                  [--local-whitening] [--regional] [--whitening]
                  [--not-pretrained] [--loss LOSS] [--loss-margin LM] 
                  [--mode MODE] [--teacher TEACHER] [--sym]
                  [--image-size N] [--neg-num N] [--query-size N]
                  [--pool-size N] [--gpu-id N] [--workers N] [--epochs N]
                  [--batch-size N] [--optimizer OPTIMIZER] [--lr LR]
                  [--momentum M] [--weight-decay W] [--print-freq N]
                  [--resume FILENAME] [--comment COMMENT] 
                  
```


To perform a symmetric test of the model that is already trained:
```bash
python test.py [-h] (--network-path NETWORK | --network-offtheshelf NETWORK)
               [--datasets DATASETS] [--image-size N] [--multiscale MULTISCALE] 
               [--whitening WHITENING] [--teacher TEACHER]
```
For the asymmetric testing: 

```bash
python test.py [-h] (--network-path NETWORK | --network-offtheshelf NETWORK)
               [--datasets DATASETS] [--image-size N] [--multiscale MULTISCALE] 
               [--whitening WHITENING] [--teacher TEACHER] [--asym]
```

Example

```bash

python test.py -npath 
                    -d 'roxford5k,rparis6k' 
                    -ms '[1, 1/2**(1/2), 1/2]' 
                    -w 'retrieval-SfM-120k' 

```


### Acknowledgements

This code is adapted and modified based on the amazing repository by F. Radenović called
[CNN Image Retrieval in PyTorch: Training and evaluating CNNs for Image Retrieval in PyTorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch)

