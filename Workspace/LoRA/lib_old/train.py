import numpy as np

import torch

from tqdm import tqdm




def get_batch_accuracy(model, data, targets, batch_size):
    output, _ = model(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    return round(acc*100,2)




def compute_val (model, task, dtype, loss, val_loader, val_loss_hist, val_acc_hist, epoch, batch_size, num_steps, device, comp_mode=False, print_shit=False) :
    model.eval()
    val_data, val_targets = next(iter(val_loader))
    val_data = val_data.to(device)
    #if task == "task_2" and not(comp_mode) :
    #    val_targets -= 5
    val_targets = val_targets.to(device)

    val_spk, val_mem = model(val_data.view(batch_size, -1))

    # LOSS
    val_loss = torch.zeros((1), dtype=dtype, device=device)
    for step in range(num_steps):
        val_loss += loss(val_mem[step], val_targets)
    val_loss_hist.append(val_loss.item())
    
    # ACCURACY
    if print_shit :
        print(f"Epoch {epoch}")
        # print(f"Train Set Loss: {train_loss_hist[epoch]:.2f}")
        print(f"Val Set Loss: {val_loss_hist[epoch]:.2f}")
        print("\n")
    val_acc_hist.append(get_batch_accuracy(model, val_data, val_targets, batch_size))
    
    
    return val_loss_hist, val_acc_hist




def train (model, task, loss, optimizer, train_loader, val_loader, num_epochs, batch_size, num_steps, device, comp_val_loader=None) :
    dtype = torch.float
    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []
    if comp_val_loader :
        comp_val_loss_hist, comp_val_acc_hist = [], []

    # Outer training loop
    for epoch in tqdm(range(num_epochs)):
        train_batch = iter(train_loader)
        loss_epoch = 0
        batch_index = 0

        # Minibatch training loop
        for i, (data, targets) in enumerate(train_batch):
            data = data.to(device)
            #if task == "task_2" :
            #    targets -= 5
            targets = targets.to(device)

            # forward pass
            model.train()
            spk_rec, mem_rec = model(data.view(batch_size, -1))
            #if i == 0 :
            #    print("batch 0 : ", mem_rec.sum(axis=0)[:2])
            #if i == 30 :
            #    print("batch 30 :", mem_rec.sum(axis=0)[:2])

            # initialize the loss & sum over time
            loss_val = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                loss_val += loss(mem_rec[step], targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            #if i == 0 :
            #    print("loss_val at batch 0 : ", loss_val.item())
            #if i == 30 :
            #    print("loss_val at batch 30 : ", loss_val.item())
            loss_epoch += loss_val.item()
            batch_index += 1

            if batch_index%50 == 0:
                print(f'{batch_index} batches used in epoch {epoch}')

        # Store loss and acc histories for future plotting
        train_loss_hist.append(loss_epoch/len(train_loader))
        train_acc_hist.append(get_batch_accuracy(model, data, targets, batch_size))

        
        # Test set
        with torch.no_grad():
            if epoch % 1 == 0:
                val_loss_hist, val_acc_hist = compute_val (model,
                                                           task,
                                                           dtype,
                                                           loss,
                                                           val_loader, 
                                                           val_loss_hist, 
                                                           val_acc_hist,
                                                           epoch,
                                                           batch_size,
                                                           num_steps,
                                                           device,
                                                           comp_mode=False,
                                                           print_shit=False)
                
                if comp_val_loader:
                    comp_val_loss_hist, comp_val_acc_hist = compute_val (model,
                                                                        task,
                                                                        dtype,
                                                                        loss,
                                                                        comp_val_loader, 
                                                                        comp_val_loss_hist, 
                                                                        comp_val_acc_hist,
                                                                        epoch,
                                                                        batch_size,
                                                                        num_steps,
                                                                        device,
                                                                        comp_mode=True,
                                                                        print_shit=False)
                    
    output = [train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist]
    if comp_val_loader :
        output += [comp_val_loss_hist, comp_val_acc_hist]
    return output