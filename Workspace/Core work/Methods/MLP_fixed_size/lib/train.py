import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from abstract_torch import get_loss, get_optimizer
import gc
import random


def compute_loss(model, num_classes, data, targets, loss, loss_name) :
    if loss_name == "CE":
        y = model(data.view(data.shape[0], -1))
        loss_val = loss(y, targets)
    #else :
    #    y = model(data.view(data.shape[0], -1))
    #    one_hot_targets = nn.functional.one_hot(targets, num_classes=num_classes).to(y.dtype)
    #    loss_val = loss(y, one_hot_targets)
    return y, loss_val


def train (model,
           loss_name, optimizer_name, lr, num_epochs,
           train_loader,
           device, random_seed,
           num_classes=10, verbose=0) :
    
    # Reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
            y, loss_val = compute_loss(model, num_classes, data, targets, loss, loss_name)
            loss_hist.append(loss_val.item())
            
            # Gradient calculation
            optimizer.zero_grad()
            loss_val.backward(create_graph=True)

            optimizer.step()


    # Clean memory
    del optimizer
    del model
    gc.collect()
    torch.cuda.empty_cache()


    return loss_hist