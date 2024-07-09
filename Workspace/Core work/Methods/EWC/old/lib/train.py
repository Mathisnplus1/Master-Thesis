import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from lib.abstract_torch import get_loss, get_optimizer


def get_number_of_neurons(model, epoch=None, batch_index=None) :
    if epoch is None or batch_index is None :
        print("In and out sizes :")
    else :
        print(f"In and out sizes on epoch {epoch}, before batch {batch_index}:")
    print("fc1 :", f"in = {model.fc1.in_features},", f"out = {model.fc1.out_features}")
    print("fc2 :", f"in = {model.fc2.in_features},", f"out = {model.fc2.out_features}")
    print("fc3 :", f"in = {model.fc3.in_features},", f"out = {model.fc3.out_features}")


def compute_loss(model, num_classes, data, targets, loss, loss_name, batch_size) :
    if loss_name == "CE":
        y = model(data.view(batch_size, -1))
        loss_val = loss(y, targets)
    else :
        y = model(data.view(batch_size, -1))
        one_hot_targets = nn.functional.one_hot(targets, num_classes=num_classes).to(y.dtype)
        loss_val = loss(y, one_hot_targets)
    return y, loss_val


def apply_mask(grad, mask):
    #print("grad :", grad.shape)
    #print("mask :", mask.shape)
    return grad * mask


# Compute second derivatives of the loss with respect to each parameter (diagonal of the Hessian)
def get_diag_hessians (layer) :
    param = layer.weight
    grad = layer.weight.grad
    second_derivative = torch.autograd.grad(
        grad,
        param,
        grad_outputs=torch.ones_like(grad),
        retain_graph=True,
        create_graph=True,
    )[0]

    return second_derivative


def get_hessian_mask(diag_hessian, percentile) :
    diag = diag_hessian.detach().cpu().numpy()
    hessian_mask = diag > np.percentile(diag, percentile)

    return hessian_mask


def get_grad_mask(layer, percentile) :
    grad = layer.weight.grad
    grad_clone = grad.clone().detach().cpu().numpy()
    grad_mask = grad_clone > np.percentile(grad_clone, percentile)

    return grad_mask


def get_overall_mask(layer, grow_from, overall_mask, overlap_mask, device):
    if grow_from == "input" :
        num_neurons = (overlap_mask.sum(axis=1) != 0).sum()
        overall_mask = torch.cat(((overall_mask*(-1*(overlap_mask-1))), torch.ones((num_neurons, layer.in_features)).to(device)), dim=0)
    else :
        num_neurons = (overlap_mask.sum(axis=0) != 0).sum()
        overall_mask = torch.cat(((overall_mask*(-1*(overlap_mask-1))), torch.ones((layer.out_features, num_neurons)).to(device)), dim=1)
    
    return overall_mask, num_neurons


def should_we_grow (loss_hist) :
    diff = np.mean(loss_hist[-40:-20]) - np.mean(loss_hist[-20])
    return 0 < diff and diff < 0.1





def train (model, grow_from, overall_masks, growth_indices,
           loss_name, optimizer_name, lr, num_epochs, batch_size,
           hessian_percentile, grad_percentile,
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

    # Get layer we want to use to perform the growth
    layer_names = ["fc1", "fc2"] if grow_from=="input" else ["fc2", "fc3"]
    layers = [getattr(model,layer_names[0]), getattr(model,layer_names[1])]

    # Initialize stuff
    loss = get_loss(loss_name)
    optimizer = get_optimizer(optimizer_name, model, lr)
    loss_hist = []
    #growth_indices = []
    delay_growth = 80
    overall_masks = [torch.tensor(overall_masks[0]).to(device), torch.tensor(overall_masks[1]).to(device)]
    layer_to_grow = "fc1" #if grow_from == "input" else "fc2"

    # Instantiate hook to freeze weights
    hook_handles = [layers[0].weight.register_hook(lambda grad: apply_mask(grad, overall_masks[0])),
                    layers[1].weight.register_hook(lambda grad: apply_mask(grad, overall_masks[1]))]

    # Epoch training loop
    for epoch in tqdm(range(num_epochs)):

        # Verbose
        if verbose > 0 :
            get_number_of_neurons(model)

        # Initialize epoch
        train_batch = iter(train_loader)
        nb_batches = len(train_batch)
        
        # Batch training loop
        for i, (data, targets) in enumerate(train_batch):

            # Initialize batch
            data = data.to(device)
            targets = targets.to(device)
            delay_growth -= 1
            model.train()

            # Forward path
            y, loss_val = compute_loss(model, num_classes, data, targets, loss, loss_name, batch_size)
            loss_hist.append(loss_val.item())
            
            # Gradient calculation
            optimizer.zero_grad()
            loss_val.backward(create_graph=True)

            # Neuro-genesis
            if len(loss_hist) in growth_indices :# delay_growth < 0 and not is_first_task and should_we_grow(loss_hist) :
                

                # FIRST GROWTH
                if layer_to_grow == "fc1" :

                    # Compute hessian mask
                    diag_hessian = get_diag_hessians(layers[0])
                    hessian_mask = get_hessian_mask(diag_hessian, hessian_percentile)

                    # Compute grad mask
                    grad_mask = get_grad_mask(layers[0], grad_percentile)

                    # Compute overlap mask
                    overlap_mask = torch.tensor((hessian_mask & grad_mask).astype(int)).to(device)

                    # Update overall mask
                    overall_masks[0], num_neurons = get_overall_mask(layers[0], grow_from, overall_masks[0], overlap_mask, device)
                    if grow_from == "input":
                        overall_masks[1] = torch.cat((overall_masks[1], torch.ones((layers[1].out_features, num_neurons)).to(device)), dim=1)

                    # Remove existing hooks
                    hook_handles[0].remove()

                    # Add new neurons
                    model.add_neurons(overlap_mask, "fc1", grow_from, device)
    

                # SECOND GROWTH


                else :
                    # Compute hessian mask
                    diag_hessian = get_diag_hessians(layers[1])
                    hessian_mask = get_hessian_mask(diag_hessian, hessian_percentile)

                    # Compute grad mask
                    grad_mask = get_grad_mask(layers[1], grad_percentile)

                    # Compute overlap mask
                    overlap_mask = torch.tensor((hessian_mask & grad_mask).astype(int)).to(device)

                    # Update overall mask
                    overall_masks[1], num_neurons = get_overall_mask(layers[1], grow_from, overall_masks[1], overlap_mask, device)
                    if grow_from != "input" :
                        overall_masks[0] = torch.cat((overall_masks[0], torch.ones((num_neurons, layers[0].in_features)).to(device)), dim=0)
                    # Remove existing hooks
                    hook_handles[1].remove()

                    # Add new neurons
                    model.add_neurons(overlap_mask, "fc2", grow_from, device)


                # END OF GROWTH

                
                # Get new layers
                layers = [getattr(model,layer_names[0]), getattr(model,layer_names[1])]

                # Instantiate new hook to freeze weights according to new overall_mask
                hook_handles = [layers[0].weight.register_hook(lambda grad: apply_mask(grad, overall_masks[0])),
                                layers[1].weight.register_hook(lambda grad: apply_mask(grad, overall_masks[1]))]

                # Reset optimizer
                optimizer = get_optimizer(optimizer_name, model, lr)

                # Get track of when growth occurs
                #growth_indices.append(len(loss_hist))
                layer_to_grow = "fc1" if layer_to_grow == "fc2" else "fc2"

                # Prevent many consecutive growths
                delay_growth = 150
                
            optimizer.step()

    return overall_masks, loss_hist, growth_indices