import torch
import torch.nn as nn
from tqdm import tqdm
from lib.test import get_batch_accuracy
from lib.abstract_torch import get_loss, get_optimizer



def count_all_parameters(model) :
    num_params = 0
    for param_name, param in model.named_parameters():
        num_param = torch.numel(param)
        print(param_name, ":", num_param)
        num_params += num_param
    return num_params


def compute_loss(model, data, targets, loss, loss_name, batch_size) :
    if loss_name == "CE":
        y = model(data.view(batch_size, -1))
        loss_val = loss(y, targets)
    else :
        y = model(data.view(batch_size, -1))
        one_hot_targets = nn.functional.one_hot(targets, num_classes=10).to(y.dtype)
        loss_val = loss(y, one_hot_targets)
    return y, loss_val


def compute_val (model, loss, loss_name, val_loader, val_loss_hist, val_acc_hist, epoch, batch_size, device, print_shit=False) :
    model.eval()
    val_data, val_targets = next(iter(val_loader))
    val_data = val_data.to(device)
    val_targets = val_targets.to(device)
    
    # Forward path
    y, val_loss_val = compute_loss(model, val_data, val_targets, loss, loss_name, batch_size)
    val_loss_hist.append(val_loss_val.item())
    
    # ACCURACY
    if print_shit :
        print(f"Epoch {epoch}")
        # print(f"Train Set Loss: {train_loss_hist[epoch]:.2f}")
        print(f"Val Set Loss: {val_loss_hist[epoch]:.2f}")
        print("\n")
    val_acc_hist.append(get_batch_accuracy(model, val_data, val_targets, batch_size))
    
    
    return val_loss_hist, val_acc_hist



def register_hooks(model, pre_layer, post_layer):
    activation = []
    grad = []

    def forward_hook(module, x, y):
        activation.append(y)
    def backward_hook(module, grad_input, grad_output) :
        grad.append(grad_output)
    
    forward_hook_handle = None
    if pre_layer :
        forward_hook_handle = pre_layer.register_forward_hook(forward_hook)
        
    backward_hook_handle = post_layer.register_full_backward_hook(backward_hook)

    return activation, grad, forward_hook_handle, backward_hook_handle




def train (model, growth_schedule, loss_name, optimizer_name, lr, train_loader, val_loader, 
           num_epochs, batch_size, device, c = 1, verbose = 0) :
    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []
    batch_index = 0
    loss = get_loss(loss_name)
    optimizer = get_optimizer(optimizer_name, model, lr)
    if growth_schedule :
        growth_schedule = iter(growth_schedule)
        hooks_list = []
    
    # Epoch training loop
    for epoch in tqdm(range(num_epochs)):
        
        # Initialize epoch
        train_batch = iter(train_loader)
        loss_epoch = 0
        
        # Batch training loop
        for i, (data, targets) in enumerate(train_batch):
            if i == 300 :
                break
            # Get data from the batch
            data = data.to(device)
            targets = targets.to(device)
            
            
            # Initialize GradMax growth
            if (growth_schedule is not None) and (batch_index%50 == 0) and (batch_index != 0):
                
                # Print the number of parameters
                if verbose > 1 :
                    count_all_parameters(model)
                
                # Remove previous hooks in case they are some
                for h in hooks_list :
                    h.remove()
                
                # Get the layer to grow and the number of neurons to add
                layer_name, num_neurons = next(growth_schedule)
                
                # Initialize the matrix on which we will perform SVD
                if layer_name == "fc1" :
                    matrix_to_SVD = torch.zeros(model.fc2.out_features, model.fc1.in_features).to(device)
                elif layer_name == "fc2" :
                    matrix_to_SVD = torch.zeros(model.fc3.out_features, model.fc2.in_features).to(device)
                    #print("matrix_to_SVD :", matrix_to_SVD.shape)

            model.train()
            
            # Get the gradient of the layer after the one we grow, for GradMax computation
            if (growth_schedule is not None) and (batch_index > 49): #and (batch_index%50 == 0) :
                if layer_name == "fc1" :
                    
                    h = data.view(batch_size, -1)
                    # register hooks
                    _, grad, _, backward_hook_handle = register_hooks(model, None, model.fc2)
                    hooks_list.append(backward_hook_handle)
                    
                    # Forward path
                    y, loss_val = compute_loss(model, data, targets, loss, loss_name, batch_size)
                    
                    # Gradient calculation + weight update
                    #output_grad = torch.zeros(torch.Size([])).requires_grad_(True).to(device)
                    optimizer.zero_grad()
                    loss_val.backward() # Pass output_grad
                    optimizer.step()
                    
                    # Compute the matrix to which we will apply SVD
                    #matrix_to_SVD += torch.mm(grad[0][0].t(),h) / batch_size
                    
                elif layer_name == "fc2" :
                    # register hooks
                    h, grad, forward_hook_handle, backward_hook_handle = register_hooks(model, model.fc1, model.fc3)
                    hooks_list.append(forward_hook_handle)
                    hooks_list.append(backward_hook_handle)
                    
                    # Forward path
                    y, loss_val = compute_loss(model, data, targets, loss, loss_name, batch_size)
                    
                    # Gradient calculation + weight update
                    #output_grad = torch.ones(torch.Size([])).requires_grad_(True)
                    optimizer.zero_grad()
                    loss_val.backward() # Pass output_grad
                    optimizer.step()
                    
                    # Compute the matrix to which we will apply SVD
                    #matrix_to_SVD += torch.mm(grad[0][0].t(),h[0]) / batch_size
            
            else :
                # Forward path
                y, loss_val = compute_loss(model, data, targets, loss, loss_name, batch_size)
                
                # Gradient calculation + weight update
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
            
            loss_epoch += loss_val.item()
            
            
            # Add neurons
            if (growth_schedule is not None) and (batch_index%50 == 0) and (batch_index != 0):
                #with torch.no_grad():
                    # Solve optimization problem (11)
                    #print("Shape of matrix_to_SVD :", matrix_to_SVD.shape)
                    # matrix_to_SVD = matrix_to_SVD.t()

                    #if layer_name == "fc1" :
                    #    model.fc1, model.fc2 = add_neurons(model.fc1, model.fc2, num_neurons, matrix_to_SVD, c, device)
                    #elif layer_name == "fc2" :
                    #    model.fc2, model.fc3 = add_neurons(model.fc2, model.fc3, num_neurons, matrix_to_SVD, c, device)
                    
                    if layer_name == "fc1" :
                        fc1_weight_grad = model.fc1.weight.grad
                        fc1_bias_grad = model.fc1.bias.grad
                        fc2_weight_grad = model.fc2.weight.grad
                        model.add_neurons(layer_name, fc1_weight_grad, fc1_bias_grad, fc2_weight_grad, num_neurons, device)
                    elif layer_name == "fc2" :
                        fc2_weight_grad = model.fc2.weight.grad
                        fc2_bias_grad = model.fc2.bias.grad
                        fc3_weight_grad = model.fc3.weight.grad
                        model.add_neurons(layer_name, fc2_weight_grad, fc2_bias_grad, fc3_weight_grad, num_neurons, device)
                    
                    optimizer = get_optimizer(optimizer_name, model, lr)
            
            # Get feedback from train and val sets
            if batch_index%50 == 49:
                with torch.no_grad():
                    # Train data
                    train_loss_hist.append(loss_epoch/len(train_loader))
                    train_acc_hist.append(get_batch_accuracy(model, data, targets, batch_size))
                    
                    # Val data
                    val_loss_hist, val_acc_hist = compute_val (model,
                                                               loss,
                                                               loss_name,
                                                               val_loader, 
                                                               val_loss_hist, 
                                                               val_acc_hist,
                                                               epoch,
                                                               batch_size,
                                                               device,
                                                               print_shit=False)
                    
            # End of batch training loop  
            batch_index += 1

            if verbose > 0 and batch_index%50 == 0:
                print(f'{i} batches used in epoch {epoch}')
        if verbose > 1 :
            print("Total number of batches :", batch_index)
    output = [train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist]
    return output