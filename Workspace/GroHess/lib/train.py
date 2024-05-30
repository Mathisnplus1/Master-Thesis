import torch
import torch.nn as nn
from tqdm import tqdm
from lib.test import get_batch_accuracy
from lib.abstract_torch import get_loss, get_optimizer


def train_ANN (model, loss_name, optimizer_name, lr, train_loader, num_epochs, batch_size, device, random_seed):

    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loss = get_loss(loss_name)
    optimizer = get_optimizer(optimizer_name, model, lr)
    model.to(device)
    for epoch in tqdm(range(num_epochs)):
        train_batch = iter(train_loader)
        for i, (data, targets) in enumerate(train_batch):
            data = data.to(device)
            targets = targets.to(device)
            model.train()
            optimizer.zero_grad()
            y = model(data.view(batch_size, -1))
            loss_val = loss(y, targets)
            loss_val.backward()
            optimizer.step()


def count_all_parameters(model) :
    num_params = 0
    for param_name, param in model.named_parameters():
        num_param = torch.numel(param)
        print(param_name, ":", num_param)
        num_params += num_param
    return num_params


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



def train (model, num_classes, growth_schedule, loss_name, optimizer_name, lr, train_loader, val_loader, 
           num_epochs, batch_size, device, init_name=None, c=1, verbose=0) :
    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []
    batch_index = 0
    loss = get_loss(loss_name)
    optimizer = get_optimizer(optimizer_name, model, lr)
    hooks_list = []

    # Initialize growth
    if (growth_schedule is not None) :
        growth_schedule = iter(growth_schedule)
        hooks_list = []
        growth_occured = False
        
        # Get the layer to grow and the number of neurons to add
        growth_epoch, growth_batch, layer_name, num_neurons = next(growth_schedule)
        
        # Initialize the matrix on which we will perform SVD
        if layer_name == "fc1" :
            growth_matrix = torch.zeros(model.fc2.out_features, model.fc1.in_features).to(device)
        elif layer_name == "fc2" :
            growth_matrix = torch.zeros(model.fc3.out_features, model.fc2.in_features).to(device)
    
    # Epoch training loop
    for epoch in tqdm(range(num_epochs)):
        
        # Initialize epoch
        train_batch = iter(train_loader)
        loss_epoch = 0
        
        # Batch training loop
        for i, (data, targets) in enumerate(train_batch):
            data = data.to(device)
            targets = targets.to(device)

            for h in hooks_list :
                h.remove()

            # Grow architecture
            if (growth_schedule is not None) and epoch == growth_epoch and i == growth_batch :
                
                if layer_name == "fc1" :
                    fc1_weight_grad = model.fc1.weight.grad
                    fc1_bias_grad = model.fc1.bias.grad
                    fc2_weight_grad = model.fc2.weight.grad
                    model.add_neurons(layer_name, fc1_weight_grad, fc1_bias_grad, fc2_weight_grad, num_neurons, device, init_name, growth_matrix, c)
                elif layer_name == "fc2" :
                    fc2_weight_grad = model.fc2.weight.grad
                    fc2_bias_grad = model.fc2.bias.grad
                    fc3_weight_grad = model.fc3.weight.grad
                    model.add_neurons(layer_name, fc2_weight_grad, fc2_bias_grad, fc3_weight_grad, num_neurons, device, init_name, growth_matrix, c)
                
                optimizer = get_optimizer(optimizer_name, model, lr)

                growth_occured = True

                # Print the number of parameters
                if verbose > 1 :
                    print("epoch :", epoch, "batch :", i)
                    count_all_parameters(model)

            model.train()
            
            # Get the gradient of the layer after the one we grow, for GradMax computation
            if (growth_schedule is not None): #and (batch_index%50 == 0) :
                if layer_name == "fc1" :
                    
                    h = data.view(batch_size, -1)
                    # register hooks
                    _, grad, _, backward_hook_handle = register_hooks(model, None, model.fc2)
                    hooks_list.append(backward_hook_handle)
                    
                    # Forward path
                    y, loss_val = compute_loss(model, num_classes, data, targets, loss, loss_name, batch_size)
                    
                    # Gradient calculation + weight update
                    optimizer.zero_grad()
                    loss_val.backward() # Pass output_grad
                    optimizer.step()
                    
                    # Compute the matrix to which we will apply SVD
                    growth_matrix += torch.mm(grad[0][0].t(),h) / batch_size
                    
                elif layer_name == "fc2" :
                    # register hooks
                    h, grad, forward_hook_handle, backward_hook_handle = register_hooks(model, model.fc1, model.fc3)
                    hooks_list.append(forward_hook_handle)
                    hooks_list.append(backward_hook_handle)
                    
                    # Forward path
                    y, loss_val = compute_loss(model, num_classes, data, targets, loss, loss_name, batch_size)
                    
                    # Gradient calculation + weight update
                    optimizer.zero_grad()
                    loss_val.backward() # Pass output_grad
                    optimizer.step()

                    # Compute the matrix to which we will apply SVD
                    growth_matrix += torch.mm(grad[0][0].t(),h[0]) / batch_size
            
            else :
                if i == 0 :
                    # register hooks
                    h, grad, forward_hook_handle, backward_hook_handle = register_hooks(model, model.fc1, model.fc3)
                    hooks_list.append(forward_hook_handle)
                    hooks_list.append(backward_hook_handle)

                # Forward path
                y, loss_val = compute_loss(model, num_classes, data, targets, loss, loss_name, batch_size)
                
                # Gradient calculation + weight update
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                if i == 0 :
                    print("grad :", grad[0][0][0])
            loss_epoch += loss_val.item()

            # Prepare next growth
            if (growth_schedule is not None) and growth_occured :
                # Remove previous hooks in case they are some
                for h in hooks_list :
                    h.remove()
                
                # Get the layer to grow and the number of neurons to add
                try :
                    growth_epoch, growth_batch, layer_name, num_neurons = next(growth_schedule)
                
                    # Initialize the matrix on which we will perform SVD
                    if layer_name == "fc1" :
                        growth_matrix = torch.zeros(model.fc2.out_features, model.fc1.in_features).to(device)
                    elif layer_name == "fc2" :
                        growth_matrix = torch.zeros(model.fc3.out_features, model.fc2.in_features).to(device)
                    growth_occured = False
                except :
                    growth_schedule = None
                    growth_occured = False

            # Get feedback from train and val sets
            if batch_index%50 == 49:
                with torch.no_grad():
                    # Train data
                    train_loss_hist.append(loss_epoch/len(train_loader))
                    train_acc_hist.append(get_batch_accuracy(model, data, targets, batch_size))
                    
                    # Val data
                    val_loss_hist, val_acc_hist = compute_val (model,
                                                               num_classes,
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
    output = [train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist]
    return output