import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from abstract_torch import get_loss, get_optimizer
import gc
import random


def get_number_of_neurons(model, epoch=None, batch_index=None) :
    if epoch is None or batch_index is None :
        print("In and out sizes :")
    else :
        print(f"In and out sizes on epoch {epoch}, before batch {batch_index}:")
    print("fc1 :", f"in = {model.fc1.in_features},", f"out = {model.fc1.out_features}")
    print("fc2 :", f"in = {model.fc2.in_features},", f"out = {model.fc2.out_features}")
    print("fc3 :", f"in = {model.fc3.in_features},", f"out = {model.fc3.out_features}")


def compute_loss(model, num_classes, data, targets, loss, loss_name) :
    if loss_name == "CE":
        y = model(data.view(data.shape[0], -1))
        loss_val = loss(y, targets)
    #else :
    #    y = model(data.view(data.shape[0], -1))
    #    one_hot_targets = nn.functional.one_hot(targets, num_classes=num_classes).to(y.dtype)
    #    loss_val = loss(y, one_hot_targets)
    return y, loss_val


def apply_mask(grad, mask):
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
    return 0 < diff and diff < 0.0002


def train (model, grow_from, hessian_masks, overall_masks, growth_record, is_first_task,
           loss_name, optimizer_name, lr, num_epochs,
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
    growth_indices = []
    delay_growth = 50
    overall_masks = [torch.tensor(overall_masks[0]).to(device), torch.tensor(overall_masks[1]).to(device)]
    layer_to_grow = "fc1" #if grow_from == "input" else "fc2"
    growth_record[0].append([])
    growth_record[1].append([])

    # Instantiate hook to freeze weights
    hook_handles = [layers[0].weight.register_hook(lambda grad: apply_mask(grad, overall_masks[0])),
                    layers[1].weight.register_hook(lambda grad: apply_mask(grad, overall_masks[1]))]

    # Epoch training loop
    for epoch in tqdm(range(int(num_epochs))):

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
            y, loss_val = compute_loss(model, num_classes, data, targets, loss, loss_name)
            loss_hist.append(loss_val.item())
            
            # Gradient calculation
            optimizer.zero_grad()
            loss_val.backward(create_graph=True)

            # Neuro-genesis
            if delay_growth < 0 and not is_first_task and should_we_grow(loss_hist) :
                
                # Keep track that growth occurs
                growth_indices += [len(loss_hist)]

                # LEFT LAYER GROWTH
                if layer_to_grow == "fc1" : 
                    # here "fc1" means "left". It is the layer fc1 if we grow from input, but fc2 if we grow from output

                    # Get hessian mask
                    hessian_mask = hessian_masks[0]

                    # Compute grad mask
                    grad_mask = get_grad_mask(layers[0], grad_percentile)

                    # Compute overlap mask
                    d1,d2 = hessian_mask.shape
                    comb = torch.tensor((hessian_mask & grad_mask[:d1,:d2]).astype(int))
                    overlap_mask = torch.cat((comb, torch.tensor(grad_mask[:d1,d2:])), dim=1)
                    overlap_mask = torch.cat((overlap_mask, torch.tensor(grad_mask[d1:])), dim=0).to(device)

                    # Update overall mask
                    overall_masks[0], num_neurons = get_overall_mask(layers[0], grow_from, overall_masks[0], overlap_mask, device)
                    if grow_from == "input":
                        overall_masks[1] = torch.cat((overall_masks[1], torch.ones((layers[1].out_features, num_neurons)).to(device)), dim=1)

                    # Remove existing hooks
                    hook_handles[0].remove()

                    # Add new neurons
                    model.add_neurons(overlap_mask, "fc1", grow_from, device)
    
                    # Keep track of growth
                    try :
                        growth_record[0][-1].append(int(layers[0].in_features.detach().cpu().numpy()))
                    except :
                        growth_record[0][-1].append(int(layers[0].in_features))
                    try :
                        growth_record[1][-1].append(int(layers[1].in_features.detach().cpu().numpy()))
                    except :
                        growth_record[1][-1].append(int(layers[1].in_features))
    

                # LEFT LAYER GROWTH
                else :
                    # here "fc1" means "left". It is the layer fc1 if we grow from input, but fc2 if we grow from output

                    # Compute hessian mask
                    hessian_mask = hessian_masks[1]

                    # Compute grad mask
                    grad_mask = get_grad_mask(layers[1], grad_percentile)

                    # Compute overlap mask
                    d1,d2 = hessian_mask.shape
                    comb = torch.tensor((hessian_mask & grad_mask[:d1,:d2]).astype(int))
                    overlap_mask = torch.cat((comb, torch.tensor(grad_mask[:d1,d2:])), dim=1)
                    overlap_mask = torch.cat((overlap_mask, torch.tensor(grad_mask[d1:])), dim=0).to(device)
                    #overlap_mask = torch.tensor((hessian_mask & grad_mask).astype(int)).to(device)

                    # Update overall mask
                    overall_masks[1], num_neurons = get_overall_mask(layers[1], grow_from, overall_masks[1], overlap_mask, device)
                    if grow_from != "input" :
                        overall_masks[0] = torch.cat((overall_masks[0], torch.ones((num_neurons, layers[0].in_features)).to(device)), dim=0)
                    # Remove existing hooks
                    hook_handles[1].remove()

                    # Add new neurons
                    model.add_neurons(overlap_mask, "fc2", grow_from, device)

                    # Keep track of growth
                    try :
                        growth_record[0][-1].append(int(layers[0].in_features.detach().cpu().numpy()))
                    except :
                        growth_record[0][-1].append(int(layers[0].in_features))
                    try :
                        growth_record[1][-1].append(int(ayers[1].in_features.detach().cpu().numpy()))
                    except :
                        growth_record[1][-1].append(int(layers[1].in_features))
    

                # END OF GROWTH

                
                # Get new layers
                layers = [getattr(model,layer_names[0]), getattr(model,layer_names[1])]

                # Instantiate new hook to freeze weights according to new overall_mask
                hook_handles = [layers[0].weight.register_hook(lambda grad: apply_mask(grad, overall_masks[0])),
                                layers[1].weight.register_hook(lambda grad: apply_mask(grad, overall_masks[1]))]

                # Reset optimizer
                optimizer = get_optimizer(optimizer_name, model, lr)

                # Get track of when growth occurs
                layer_to_grow = "fc1" if layer_to_grow == "fc2" else "fc2"

                # Prevent many consecutive growths
                delay_growth = 50

            optimizer.step()


    # COMPUTE HESSIAN MASK

    hessian_masks = []
    optimizer.zero_grad()

    # Get 100% of the batches
    train_batch = iter(train_loader)
    nb_batches = len(train_batch)
    sample_size = max(1, nb_batches // 1) # divide by 10 for 10%
    random_batches = random.sample(list(train_batch), sample_size)

    # Stack the gradients computed on each batch
    for i, (data, targets) in enumerate(random_batches):
        data = data.to(device)
        targets = targets.to(device)
        _, loss_val = compute_loss(model, num_classes, data, targets, loss, loss_name)
        loss_val.backward(create_graph=True)

    # Compute hessian masks (which relies on the gradients computed above)
    diag_hessian = get_diag_hessians(layers[0])
    hessian_mask_fc1 = get_hessian_mask(diag_hessian, hessian_percentile)
    hessian_masks.append(hessian_mask_fc1)

    diag_hessian = get_diag_hessians(layers[1])
    hessian_mask_fc2 = get_hessian_mask(diag_hessian, hessian_percentile)
    hessian_masks.append(hessian_mask_fc2)


    # Clean memory
    del optimizer
    del model
    gc.collect()
    torch.cuda.empty_cache()


    return hessian_masks, overall_masks, growth_record, loss_hist, growth_indices