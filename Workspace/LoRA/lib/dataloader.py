import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import torchvision.transforms as T
import numpy as np


MNIST_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
CIFAR_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



def get_task_loaders(data_path, batch_size, random_seed, train_percentage=0.7, download=False) :
    # create pMNIST dataset
    rnd = np.random.RandomState(random_seed)
    idx_permute = torch.from_numpy(rnd.permutation(28*28))
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,)),
        T.Lambda(lambda x: torch.flatten(x)[idx_permute]) # added permutation
    ])
    
    train_pmnist = datasets.MNIST(data_path, train=True, download=download, transform=transform)
    test_pmnist = datasets.MNIST(data_path, train=False, download=download, transform=transform)

    # create a mask to filter indices for each label
    train_mask = torch.tensor([label in range(10) for label in train_pmnist.targets])
    test_mask = torch.tensor([label in range(10) for label in test_pmnist.targets])

    # Create Subset datasets for train, validation, and test
    train_dataset = Subset(train_pmnist, torch.where(train_mask)[0])
    test_dataset = Subset(test_pmnist, torch.where(test_mask)[0])

    # split train into train & validation
    train_size = int(train_percentage * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader



def get_MNIST_loaders(path, batch_size, random_seed, train_percentage=0.7, download=False) :

    train_loaders_list, val_loaders_list, test_loaders_list = [], [], []

    # LOAD MNIST 
    mnist_train = datasets.MNIST(root=path, train=True, download=download, transform=MNIST_transform)
    mnist_test = datasets.MNIST(root=path, train=False, download=download, transform=MNIST_transform)

    # SPLIT DATA
    train_dataset, val_dataset = random_split(mnist_train, [train_percentage, 1-train_percentage], generator=torch.Generator().manual_seed(random_seed))
    test_mask = torch.tensor([label in range(10) for label in mnist_test.targets])
    test_dataset = Subset(mnist_test, torch.where(test_mask)[0])

    # GET LOADERS
    lists_of_labels = [range(5), [0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    for label_list in lists_of_labels :   
        # train/val/test_dataset's type is Subset, so can't directly access its targets through .targets
        train_subset_indices = train_dataset.indices
        train_subset_targets = [train_dataset.dataset.targets[i] for i in train_subset_indices]
        val_subset_indices = val_dataset.indices
        val_subset_targets = [val_dataset.dataset.targets[i] for i in val_subset_indices]
        test_subset_indices = test_dataset.indices
        test_subset_targets = [test_dataset.dataset.targets[i] for i in test_subset_indices]
        # create a mask to filter indices for each label
        train_mask = torch.tensor([label in label_list for label in train_subset_targets])
        val_mask = torch.tensor([label in label_list for label in val_subset_targets])
        test_mask = torch.tensor([label in label_list for label in test_subset_targets])
        # create Subset datasets
        train_subset = Subset(train_dataset, torch.where(train_mask)[0])
        val_subset = Subset(val_dataset, torch.where(val_mask)[0])
        test_subset = Subset(test_dataset, torch.where(test_mask)[0])
        # get loaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, drop_last=True)
        # store loaders in the lists of loaders
        train_loaders_list.append(train_loader)
        val_loaders_list.append(val_loader)
        test_loaders_list.append(test_loader)

    return train_loaders_list, val_loaders_list, test_loaders_list

def get_FMNIST_loaders(path, class_names, batch_size, download=False) :

    # load MNIST 
    fmnist_train = datasets.FashionMNIST(root=path, train=True, download=download, transform=MNIST_transform)
    fmnist_test = datasets.FashionMNIST(root=path, train=False, download=download, transform=MNIST_transform)


    # create a mask to filter indices for each label
    train_mask = torch.tensor([label in class_names for label in fmnist_train.targets])
    test_mask = torch.tensor([label in class_names for label in fmnist_test.targets])

    # Create Subset datasets for train, validation, and test
    train_dataset = Subset(fmnist_train, torch.where(train_mask)[0])
    test_dataset = Subset(fmnist_test, torch.where(test_mask)[0])

    # split train into train & validation
    train_size = int(0.7 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader


def get_CIFAR10_loaders(path, class_names, batch_size, download=False) :

    # load MNIST 
    cifar10_train = datasets.CIFAR10(root=path, train=True, download=download, transform=CIFAR_transform)
    cifar10_test = datasets.CIFAR10(root=path, train=False, download=download, transform=CIFAR_transform)

    # create a mask to filter indices for each label
    train_mask = torch.tensor([label in class_names for label in cifar10_train.targets])
    test_mask = torch.tensor([label in class_names for label in cifar10_test.targets])

    # Create Subset datasets for train, validation, and test
    train_dataset = Subset(cifar10_train, torch.where(train_mask)[0])
    test_dataset = Subset(cifar10_test, torch.where(test_mask)[0])

    # split train into train & validation
    train_size = int(0.7 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader


def get_CIFAR100_loaders(path, class_names, batch_size, download=False) :

    # load MNIST 
    cifar100_train = datasets.CIFAR100(root=path, train=True, download=download, transform=CIFAR_transform)
    cifar100_test = datasets.CIFAR100(root=path, train=False, download=download, transform=CIFAR_transform)

    # create a mask to filter indices for each label
    train_mask = torch.tensor([label in class_names for label in cifar100_train.targets])
    test_mask = torch.tensor([label in class_names for label in cifar100_test.targets])

    # Create Subset datasets for train, validation, and test
    train_dataset = Subset(cifar100_train, torch.where(train_mask)[0])
    test_dataset = Subset(cifar100_test, torch.where(test_mask)[0])

    # split train into train & validation
    train_size = int(0.7 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader