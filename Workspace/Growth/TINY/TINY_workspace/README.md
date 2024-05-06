# TINYpub

## Description
Implementation of the architecture growth method "TINY" (https://openreview.net/forum?id=Qp33jnRKda) on three different type of neural network architectures :
* Linear layer (MLP)
* Convolutional layer (CL)
* ResNet block (ResNet)


## Installation
The dependancies and theirs versions are in pyproject.toml. 
You can either execute this file or install them by hand checking the version with pyproject.toml.

To execute pyproject.toml, create a conda environnement, install the module poetry then run the command :
```
poetry install
```
### Working with CPUs only
Dowmload a version of torch and torchvision
```
conda install pytorch==2.2.0 torchvision==0.17.0 -c pytorch
```
### Working with GPU(s) :
Dowmload a version of torch and torchvision
```
# CUDA 11.8
conda install pytorch==2.2.0 torchvision==0.17.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.2.0 torchvision==0.17.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
The notebooks are to be updated ...
