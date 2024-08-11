from lib.train_old_thresholding import train as train_model
from lib.models import ANN
import numpy as np

def initialize_model(method_settings, global_seed) :
    try :
        num_inputs = method_settings["num_inputs"]
        num_hidden_root = method_settings["num_hidden_root"]
        num_outputs = method_settings["num_outputs"]
    except ValueError:
        print("One or more of the required settings to initialize the model are missing. Please check the method settings.")
    
    model = ANN(num_inputs, num_hidden_root, num_outputs, global_seed)

    return model

def initialize_training(model, method_settings, benchmark_settings=None, device=None) :
    try :
        grow_from = method_settings["grow_from"]
    except ValueError:
        print("One or more of the required settings to initialize the HPO are missing. Please check the method settings.")
    if grow_from == "input" :
        overall_masks = [np.ones_like(model.fc1.weight.data.cpu().numpy()),
                        np.ones_like(model.fc2.weight.data.cpu().numpy())]
    else :
        overall_masks = [np.ones_like(model.fc2.weight.data.cpu().numpy()),
                        np.ones_like(model.fc3.weight.data.cpu().numpy())]
    return None, overall_masks

def train (model, method_settings, params, HPs, train_loader, device, global_seed, verbose=0) :
    # Get method settings
    try :
        grow_from = method_settings["grow_from"]
        hessian_percentile = method_settings["hessian_percentile"]
        grad_percentile = method_settings["grad_percentile"]
        loss_name = method_settings["loss_name"]
        optimizer_name = method_settings["optimizer_name"]
    except ValueError:
        print("One or more of the required settings to train the model are missing. Please check the method settings.")

    # Get params
    try :
        hessian_masks = params["hessian_masks"]
        overall_masks = params["overall_masks"]
        is_first_task = params["is_first_task"]
    except ValueError:
        print("One or more of the required params to train the model are missing. Please check the params.")

    # Get HPs
    try :
        lr = HPs["lr"]
        num_epochs = HPs["num_epochs"]
    except ValueError:
        print("One or more of the required HPs to train the model are missing. Please check the HPs.")

    # Train
    hessian_masks, overall_masks, loss_hist, growth_indices = train_model(model, grow_from, hessian_masks, overall_masks, is_first_task,
                      loss_name, optimizer_name, lr, num_epochs,
                      hessian_percentile, grad_percentile,
                      train_loader,
                      device, global_seed, 
                      verbose=verbose)
    
    return hessian_masks, overall_masks, loss_hist, growth_indices
            
    