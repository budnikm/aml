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
Most parameters are the same as in [CNN Image Retrieval in PyTorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch). Here, we describe parameters added or modified in this work, namely:  
--arch - architecture of the model to be trained, in our case the student.  
--mode - is the training mode, which determines how the dataset is handled, e.g. are the tuples constructed randomly or with mining; which examples are coming from the teacher vs student, etc. So for example while the --loss is set to 'contrastive', 'ts' enables standard student-teacher training (includes mining), 'ts_self' trains using the Contr+ approach, 'reg' uses the regression. When using 'rand' or 'reg' no mining is used. With 'std' it follows the original training protocol from [here](https://github.com/filipradenovic/cnnimageretrieval-pytorch) (the teacher model is not used).  
--teacher - the model of the teacher(vgg16 or resnet101), note that this param makes the last layer of the student match that of the teacher. Therefore, this can be used even in a standard symmetric training.  
--sym - a flag that indicates if the training should be symmetric or asymmetric.  

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

Examples:

Perform a symmetric test with a pre-trained model:

```bash

python test.py -npath  mobilenet-v2-gem-contr-vgg16 -d 'roxford5k,rparis6k' -ms '[1, 1/2**(1/2), 1/2]' -w retrieval-SfM-120k --teacher vgg16
```

For an asymmetric test:

```bash

python test.py -npath  mobilenet-v2-gem-contr-vgg16 -d 'roxford5k,rparis6k' -ms '[1, 1/2**(1/2), 1/2]' -w retrieval-SfM-120k --teacher vgg16 --asym
```

If you are interested in just the trained models, you can find the links to them in the test.py file. 

### Acknowledgements

This code is adapted and modified based on the amazing repository by F. Radenović called
[CNN Image Retrieval in PyTorch: Training and evaluating CNNs for Image Retrieval in PyTorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch)

