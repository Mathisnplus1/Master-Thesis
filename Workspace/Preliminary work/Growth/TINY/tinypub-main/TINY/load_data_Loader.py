#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms as tt
from torchvision.transforms import ToTensor, Normalize


def my_one_hot100(y):
    b = torch.zeros(100)
    b[y] = torch.ones(1)
    return (b)


def my_one_hot(y):
    b = torch.zeros(10)
    b[y] = torch.ones(1)
    return b


def data_loader(tr, te, batch_size=100):
    train_dataloader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(te, batch_size=batch_size, shuffle=True)
    return (train_dataloader, test_dataloader)


def load_database_CIFAR100(AugD=True):
    global X_train_rescale
    global Y_train_rescale

    X_train_rescale = torch.zeros(0, 3, 32, 32)
    Y_train_rescale = torch.zeros(0, 100)

    # my_mean = torch.tensor([0.5068, 0.4864, 0.4407])
    # my_std = torch.tensor([0.2675, 0.2565, 0.2762])

    stats = ((0.5068, 0.4864, 0.4407), (0.2675, 0.2565, 0.2762))
    if AugD:
        train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                                 tt.RandomHorizontalFlip(),
                                 tt.ToTensor(),
                                 tt.Normalize(*stats, inplace=True)
                                 ])

    else:
        train_tfms = tt.Compose([
            tt.ToTensor(),
            tt.Normalize(*stats, inplace=True)
        ])

    valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)
                             ])

    training_data = datasets.CIFAR100(
        root="data",
        train=True,
        download=True,
        transform=train_tfms,
        target_transform=my_one_hot100
    )

    test_data = datasets.CIFAR100(
        root="data",
        train=False,
        download=True,
        transform=valid_tfms,
        target_transform=my_one_hot100
    )

    return (training_data, test_data)

    # train_dataloader_100 = DataLoader(training_data, batch_size=100, shuffle=True)
    # test_dataloader_100 = DataLoader(test_data, batch_size=100, shuffle=True)

    # train_dataloader_1000 = DataLoader(training_data, batch_size=1000, shuffle=True)
    # test_dataloader_1000 = DataLoader(test_data, batch_size=1000, shuffle=True)

    # return({100:(train_dataloader_100, test_dataloader_100),
    #        1000 : (train_dataloader_1000, test_dataloader_1000)})


def load_database_CIFAR10(AugD=True):
    global X_train_rescale
    global Y_train_rescale

    X_train_rescale = torch.zeros(0, 3, 32, 32)
    Y_train_rescale = torch.zeros(0, 100)

    # my_mean = torch.tensor([0.5068, 0.4864, 0.4407])
    # my_std = torch.tensor([0.2675, 0.2565, 0.2762])

    stats = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    if AugD:
        train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                                 tt.RandomHorizontalFlip(),
                                 tt.ToTensor(),
                                 tt.Normalize(*stats, inplace=True)
                                 ])

    else:
        train_tfms = tt.Compose([
            tt.ToTensor(),
            tt.Normalize(*stats, inplace=True)
        ])

    valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)
                             ])

    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=train_tfms,
        target_transform=my_one_hot
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=valid_tfms,
        target_transform=my_one_hot
    )

    return (training_data, test_data)

    # train_dataloader_100 = DataLoader(training_data, batch_size=100, shuffle=True)
    # test_dataloader_100 = DataLoader(test_data, batch_size=100, shuffle=True)

    # train_dataloader_1000 = DataLoader(training_data, batch_size=1000, shuffle=True)
    # test_dataloader_1000 = DataLoader(test_data, batch_size=1000, shuffle=True)

    # return({100:(train_dataloader_100, test_dataloader_100),
    #        1000 : (train_dataloader_1000, test_dataloader_1000)})


'''
def load_database_CIFAR10(batch_size = 200) :
    global X_train_rescale 
    global Y_train_rescale
    
    X_train_rescale = torch.zeros(0, 3, 32, 32)
    Y_train_rescale = torch.zeros(0, 10)
    
    #my_mean = torch.tensor([0.5068, 0.4864, 0.4407])
    #my_std = torch.tensor([0.2675, 0.2565, 0.2762])
    
    
    stats = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_tfms = tt.Compose([tt.RandomCrop(32, padding=4,padding_mode='reflect'), 
                         tt.RandomHorizontalFlip(), 
                         tt.ToTensor(), 
                         tt.Normalize(*stats,inplace=True)
                        ])
    valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)
                        ])
    
    training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform= train_tfms,
    target_transform = my_one_hot
    )
    
    test_data = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform= valid_tfms,
            target_transform = my_one_hot
        ) 
    
    
    return(training_data, test_data)
    
    train_dataloader_100 = DataLoader(training_data, batch_size=100, shuffle=True)
    test_dataloader_100 = DataLoader(test_data, batch_size=100, shuffle=True)
    
    train_dataloader_1000 = DataLoader(training_data, batch_size=1000, shuffle=True)
    test_dataloader_1000 = DataLoader(test_data, batch_size=1000, shuffle=True)
    return({100:(train_dataloader_100, test_dataloader_100), 
            1000 : (train_dataloader_1000, test_dataloader_1000)})

    
        
'''


def load_database_MNIST(batch_size=200):
    my_mean = torch.tensor(0.1307)
    my_std = torch.tensor(0.3081)

    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.Compose([ToTensor(), Normalize(my_mean, my_std)]),
        target_transform=my_one_hot
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.Compose([ToTensor(), Normalize(my_mean, my_std)]),
        target_transform=my_one_hot
    )

    # return(training_data, test_data)
    return (training_data, test_data)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return (train_dataloader, test_dataloader)
