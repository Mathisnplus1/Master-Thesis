from lib.train import train as train_model
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


def train (model, method_settings, params, HPs, train_loader, device, global_seed, verbose=0) :
    # Get method settings
    try :
        loss_name = method_settings["loss_name"]
        optimizer_name = method_settings["optimizer_name"]
    except ValueError:
        print("One or more of the required settings to train the model are missing. Please check the method settings.")

    # Get HPs
    try :
        lr = HPs["lr"]
        num_epochs = HPs["num_epochs"]
    except ValueError:
        print("One or more of the required HPs to train the model are missing. Please check the HPs.")

    # Train
    _ = train_model(model,
                      loss_name, optimizer_name, lr, num_epochs,
                      train_loader,
                      device, global_seed, verbose=verbose)

            
    