import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import torchvision.transforms as T
import numpy as np



def get_task_loaders(data_path, batch_size, global_seed, random_seed, train_percentage=0.7, difficulty="standard", download=False, transform=None) :

    # Reproducibility
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)

    # create pMNIST dataset
    rnd = np.random.RandomState(random_seed)
    if difficulty == "easy" :
        idx_permute = torch.from_numpy(rnd.permutation(8*8))
        def permute (x) :
            x = x.view(28,28)
            x[10:18,10:18] = torch.flatten(x[10:18,10:18])[idx_permute].view(8,8)
            return x
    else :
        idx_permute = torch.from_numpy(rnd.permutation(28*28))
        def permute (x) :
            x = torch.flatten(x)[idx_permute]
            return x
     
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,)),
        T.Lambda(lambda x: permute(x))
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader