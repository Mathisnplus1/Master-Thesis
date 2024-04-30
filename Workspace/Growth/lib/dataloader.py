import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


def get_MNIST_loaders(path, class_names, batch_size) :

    # load MNIST 
    mnist_train = datasets.MNIST(root=path, train=True, download=False, transform=transform)
    mnist_test = datasets.MNIST(root=path, train=False, download=False, transform=transform)


    # create a mask to filter indices for each label
    train_mask = torch.tensor([label in class_names for label in mnist_train.targets])
    test_mask = torch.tensor([label in class_names for label in mnist_test.targets])

    # Create Subset datasets for train, validation, and test
    train_dataset = Subset(mnist_train, torch.where(train_mask)[0])
    test_dataset = Subset(mnist_test, torch.where(test_mask)[0])

    # split train into train & validation
    train_size = int(0.7 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader

def get_FMNIST_loaders(path, class_names, batch_size) :

    # load MNIST 
    fmnist_train = datasets.FashionMNIST(root=path, train=True, download=False, transform=transform)
    fmnist_test = datasets.FashionMNIST(root=path, train=False, download=False, transform=transform)


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