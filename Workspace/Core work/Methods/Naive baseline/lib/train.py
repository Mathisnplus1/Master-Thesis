import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from lib.test import get_batch_accuracy
from lib.abstract_torch import get_loss, get_optimizer




def compute_loss(model, num_classes, data, targets, loss, loss_name, batch_size) :
    if loss_name == "CE":
        y = model(data.view(batch_size, -1))
        loss_val = loss(y, targets)
    else :
        y = model(data.view(batch_size, -1))
        one_hot_targets = nn.functional.one_hot(targets, num_classes=num_classes).to(y.dtype)
        loss_val = loss(y, one_hot_targets)
    return y, loss_val



def compute_val (model, num_classes, loss, loss_name, val_loader, val_loss_hist, val_acc_hist, epoch, batch_size, device, print_shit=False) :
    model.eval()
    val_data, val_targets = next(iter(val_loader))
    val_data = val_data.to(device)
    val_targets = val_targets.to(device)
    
    # Forward path
    y, val_loss_val = compute_loss(model, num_classes, val_data, val_targets, loss, loss_name, batch_size)
    val_loss_hist.append(val_loss_val.item())
    
    # ACCURACY
    if print_shit :
        print(f"Epoch {epoch}")
        # print(f"Train Set Loss: {train_loss_hist[epoch]:.2f}")
        print(f"Val Set Loss: {val_loss_hist[epoch]:.2f}")
        print("\n")
    val_acc_hist.append(get_batch_accuracy(model, val_data, val_targets, batch_size))
    
    return val_loss_hist, val_acc_hist



def train (model, 
           loss_name, optimizer_name, lr, num_epochs, batch_size,
           train_loader, 
           device, random_seed, num_classes=10, verbose=0) :

    # Reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Add neurons for the task at hand
    model.add_neurons()
    
    # Freeze neurons trained on the previous tasks
    model.freeze_neurons()

    # Send model to device
    model.to(device)

    # Verbose 
    if verbose > 0 :
        print("Number of frozen neurons :", np.sum([layer.out_features for layer in model.fc1s[:-1]]))
        print("Number of trainable neurons :", model.fc1s[-1].out_features)

    # Initialize stuff
    loss = get_loss(loss_name)
    optimizer = get_optimizer(optimizer_name, model, lr)
    loss_hist = []
    
    # Epoch training loop
    for epoch in tqdm(range(num_epochs)):

        # Initialize epoch
        train_batch = iter(train_loader)
        nb_batches = len(train_batch)
        
        # Batch training loop
        for i, (data, targets) in enumerate(train_batch):

            # Initialize batch
            data = data.to(device)
            targets = targets.to(device)
            model.train()

            # Forward path
            y, loss_val = compute_loss(model, num_classes, data, targets, loss, loss_name, batch_size)
            loss_hist.append(loss_val.item())
            
            # Gradient calculation
            optimizer.zero_grad()
            loss_val.backward(create_graph=True)
            optimizer.step()

    return loss_hist